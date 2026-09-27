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

void UOrganicCameraComponent::OnOscMessage(const FOSCMessage& Message, const FString& IPAddress, int32 Port)
{
	float Yaw = 0.f, Pitch = 0.f;
	if (UOSCManager::GetFloat(Message, 0, Yaw) && UOSCManager::GetFloat(Message, 1, Pitch))
	{
		FScopeLock Lock(&ResidualLock);
		LatestAIResidual = FRotator(Pitch, Yaw, 0.f);
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
}

FRotator UOrganicCameraComponent::ComputeResidual(float DeltaTime)
{
	switch (Mode)
	{
	case EOrganicCameraMode::Perlin:
		return FRotator::ZeroRotator;  // Perlin sets the offset directly in TickComponent
	case EOrganicCameraMode::AI:
	{
		FScopeLock Lock(&ResidualLock);
		// Stale data (Python stopped) -> no residual instead of repeating the last one.
		if (FPlatformTime::Seconds() - LastAIMessageTime > 0.2)
		{
			return FRotator::ZeroRotator;
		}
		const FRotator R = LatestAIResidual;
		LatestAIResidual = FRotator::ZeroRotator;  // each residual is applied once
		return R;
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
	const float SpeedMs = Pawn->GetVelocity().Size2D() / 100.f;  // cm/s -> m/s

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
		Offset *= (1.f - Leak);
	}
	Offset.Yaw = FMath::Clamp(Offset.Yaw, -MaxOffsetDeg, MaxOffsetDeg);
	Offset.Pitch = FMath::Clamp(Offset.Pitch, -MaxOffsetDeg, MaxOffsetDeg);
	Offset.Roll = 0.f;

	Camera->SetWorldRotation(Control + Offset);
}
