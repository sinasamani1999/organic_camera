#include "OrganicCameraComponent.h"

#include "Camera/CameraComponent.h"
#include "Camera/CameraShakeBase.h"
#include "Camera/PlayerCameraManager.h"
#include "GameFramework/Character.h"
#include "GameFramework/PlayerController.h"
#include "InputCoreTypes.h"
#include "OSCClient.h"
#include "OSCManager.h"
#include "OSCServer.h"
#include "HAL/FileManager.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

UOrganicCameraComponent::UOrganicCameraComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	// Run after the controller has applied mouse input, before the camera view is read.
	PrimaryComponentTick.TickGroup = TG_PostPhysics;
}

void UOrganicCameraComponent::BeginPlay()
{
	Super::BeginPlay();

	Camera = GetOwner()->FindComponentByClass<UCameraComponent>();
	if (Camera)
	{
		// The component sets the camera rotation itself (control rotation + organic offset),
		// so aiming and projectiles keep using the clean control rotation.
		Camera->bUsePawnControlRotation = false;
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("OrganicCamera: owner has no CameraComponent"));
	}

	if (APawn* Pawn = Cast<APawn>(GetOwner()))
	{
		LastControlRotation = Pawn->GetControlRotation();
	}

	OscClient = UOSCManager::CreateOSCClient(TEXT("127.0.0.1"), SendPort, TEXT("OrganicCameraClient"), this);
	OscServer = UOSCManager::CreateOSCServer(TEXT("127.0.0.1"), ReceivePort, false, true, TEXT("OrganicCameraServer"), this);
	if (OscServer)
	{
		OscServer->OnOscMessageReceived.AddDynamic(this, &UOrganicCameraComponent::OnOscMessage);
	}
	UE_LOG(LogTemp, Log, TEXT("OrganicCamera: sending /oc/state to %d, listening /oc/residual on %d"), SendPort, ReceivePort);

	SetMode(Mode);
}

void UOrganicCameraComponent::EndPlay(const EEndPlayReason::Type Reason)
{
	StopOffsetLog();
	if (OscServer)
	{
		OscServer->Stop();
	}
	Super::EndPlay(Reason);
}

void UOrganicCameraComponent::SetMode(EOrganicCameraMode NewMode)
{
	Mode = NewMode;
	Offset = FRotator::ZeroRotator;

	APawn* Pawn = Cast<APawn>(GetOwner());
	APlayerController* PC = Pawn ? Cast<APlayerController>(Pawn->GetController()) : nullptr;
	if (PC && PC->PlayerCameraManager)
	{
		PC->PlayerCameraManager->StopAllCameraShakes(true);
		if (Mode == EOrganicCameraMode::CameraShake)
		{
			if (ShakeClass)
			{
				PC->ClientStartCameraShake(ShakeClass, ShakeScale);
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("OrganicCamera: CameraShake mode but ShakeClass is not set"));
			}
		}
	}
	UE_LOG(LogTemp, Log, TEXT("OrganicCamera: mode = %s"), *UEnum::GetValueAsString(Mode));
}

void UOrganicCameraComponent::ResetLayer()
{
	Time = 0.f;
	StepRate = FRotator::ZeroRotator;
	StepTimeLeft = 0.f;
	{
		FScopeLock Lock(&ResidualLock);
		LatestAIResidual = FRotator::ZeroRotator;
		bNewAIResidual = false;
	}
	SetMode(Mode);  // zeroes the offset and restarts the camera shake
}

void UOrganicCameraComponent::StartOffsetLog(const FString& Label)
{
	StopOffsetLog();
	bLogging = true;
	LogLabel = Label;
	LogTime = 0.0;
	LogLines.Reset();
	if (APawn* P = Cast<APawn>(GetOwner()))
	{
		PrevControlForLog = P->GetControlRotation();
	}
	LogLines.Add(TEXT("time,frame,mode,control_pitch,control_yaw,offset_pitch,offset_yaw,view_pitch,view_yaw,speed"));
	UE_LOG(LogTemp, Log, TEXT("OrganicCamera: offset log started (%s)"), *Label);
}

void UOrganicCameraComponent::StopOffsetLog()
{
	if (!bLogging)
	{
		return;
	}
	bLogging = false;
	const FString Dir = FPaths::ProjectSavedDir() / TEXT("OrganicCamera") / TEXT("Logs");
	IFileManager::Get().MakeDirectory(*Dir, true);
	const FString ModeName = UEnum::GetValueAsString(Mode).Replace(TEXT("EOrganicCameraMode::"), TEXT(""));
	const FString Path = Dir / FString::Printf(TEXT("%s_%s_%s.csv"), *LogLabel, *ModeName, *FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S")));
	FFileHelper::SaveStringArrayToFile(LogLines, *Path);
	UE_LOG(LogTemp, Log, TEXT("OrganicCamera: offset log (%d rows) saved to %s"), LogLines.Num() - 1, *Path);
	LogLines.Reset();
}

void UOrganicCameraComponent::OnOscMessage(const FOSCMessage& Message, const FString& IPAddress, int32 Port)
{
	float Yaw = 0.f, Pitch = 0.f;
	if (UOSCManager::GetFloat(Message, 0, Yaw) && UOSCManager::GetFloat(Message, 1, Pitch))
	{
		FScopeLock Lock(&ResidualLock);
		LatestAIResidual = FRotator(Pitch, Yaw, 0.f);
		bNewAIResidual = true;
		LastAIMessageTime = FPlatformTime::Seconds();
	}
}

void UOrganicCameraComponent::HandleModeKeys()
{
	APawn* Pawn = Cast<APawn>(GetOwner());
	APlayerController* PC = Pawn ? Cast<APlayerController>(Pawn->GetController()) : nullptr;
	if (!PC)
	{
		return;
	}
	if (PC->WasInputKeyJustPressed(EKeys::One))   SetMode(EOrganicCameraMode::Off);
	if (PC->WasInputKeyJustPressed(EKeys::Two))   SetMode(EOrganicCameraMode::Perlin);
	if (PC->WasInputKeyJustPressed(EKeys::Three)) SetMode(EOrganicCameraMode::CameraShake);
	if (PC->WasInputKeyJustPressed(EKeys::Four))  SetMode(EOrganicCameraMode::AI);
	if (PC->WasInputKeyJustPressed(EKeys::L))
	{
		if (bLogging) StopOffsetLog(); else StartOffsetLog(TEXT("manual"));
	}
}

FRotator UOrganicCameraComponent::ComputeResidual(float DeltaTime)
{
	switch (Mode)
	{
	case EOrganicCameraMode::Perlin:
		return FRotator::ZeroRotator;  // Perlin sets the offset directly in TickComponent
	case EOrganicCameraMode::AI:
	{
		{
			FScopeLock Lock(&ResidualLock);
			// Stale data (Python stopped) -> stop adding residuals.
			if (FPlatformTime::Seconds() - LastAIMessageTime > 0.2)
			{
				StepTimeLeft = 0.f;
				bNewAIResidual = false;
				return FRotator::ZeroRotator;
			}
			if (bNewAIResidual)
			{
				// Whatever is left of the previous step is added to the new one, so no rotation is lost.
				const FRotator Leftover = StepRate * StepTimeLeft;
				StepRate = (LatestAIResidual + Leftover) * (1.f / ModelStepSeconds);
				StepTimeLeft = ModelStepSeconds;
				bNewAIResidual = false;
			}
		}
		// Apply the residual as a constant rate over one model step (smooth at any render frame rate).
		const float Dt = FMath::Min(DeltaTime, StepTimeLeft);
		StepTimeLeft -= Dt;
		return StepRate * Dt;
	}
	default:
		return FRotator::ZeroRotator;  // Off and CameraShake (handled by the camera manager)
	}
}

void UOrganicCameraComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	Time += DeltaTime;
	++FrameIndex;
	HandleModeKeys();

	APawn* Pawn = Cast<APawn>(GetOwner());
	if (!Pawn || !Camera)
	{
		return;
	}

	// Player intent this frame = change of the control rotation (mouse), degrees.
	const FRotator Control = Pawn->GetControlRotation();
	const FRotator Intent = (Control - LastControlRotation).GetNormalized();
	LastControlRotation = Control;
	const float SpeedMs = SpeedOverride >= 0.f ? SpeedOverride : Pawn->GetVelocity().Size2D() / 100.f;  // m/s

	if (OscClient)
	{
		FOSCMessage Msg;
		UOSCManager::SetOSCMessageAddress(Msg, UOSCManager::ConvertStringToOSCAddress(TEXT("/oc/state")));
		UOSCManager::AddInt32(Msg, static_cast<int32>(FrameIndex));
		UOSCManager::AddFloat(Msg, DeltaTime);
		UOSCManager::AddFloat(Msg, Intent.Yaw);
		UOSCManager::AddFloat(Msg, Intent.Pitch);
		UOSCManager::AddFloat(Msg, SpeedMs);
		OscClient->SendOSCMessage(Msg);
	}

	if (Mode == EOrganicCameraMode::Perlin)
	{
		// Classic procedural camera noise: the offset IS the noise value (no accumulation).
		const float T = Time * PerlinFrequencyHz;
		Offset = FRotator(PerlinAmplitudeDeg * FMath::PerlinNoise1D(T + 57.3f),
		                  PerlinAmplitudeDeg * FMath::PerlinNoise1D(T), 0.f);
	}
	else
	{
		// Accumulate the organic residual into a bounded, slowly re-centring offset.
		Offset += ComputeResidual(DeltaTime);
		Offset *= FMath::Exp(-LeakPerSecond * DeltaTime);
	}
	Offset.Yaw = FMath::Clamp(Offset.Yaw, -MaxOffsetDeg, MaxOffsetDeg);
	Offset.Pitch = FMath::Clamp(Offset.Pitch, -MaxOffsetDeg, MaxOffsetDeg);
	Offset.Roll = 0.f;

	Camera->SetWorldRotation(Control + Offset);

	if (bLogging)
	{
		// view = what the player camera manager rendered last frame (includes Camera Shake), relative to control
		FRotator View = Control + Offset;
		APlayerController* PC = Cast<APlayerController>(Pawn->GetController());
		if (PC && PC->PlayerCameraManager)
		{
			View = PC->PlayerCameraManager->GetCameraRotation();
		}
		// GetCameraRotation() is the view rendered last frame, so compare it with last frame's control rotation;
		// comparing with this frame's control would count the player's own turning as camera shake.
		const FRotator ViewRel = (View - PrevControlForLog).GetNormalized();
		PrevControlForLog = Control;
		LogLines.Add(FString::Printf(TEXT("%.5f,%lld,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f"), LogTime, FrameIndex,
			static_cast<int32>(Mode), Control.Pitch, Control.Yaw, Offset.Pitch, Offset.Yaw, ViewRel.Pitch, ViewRel.Yaw, SpeedMs));
		LogTime += DeltaTime;
	}
}
