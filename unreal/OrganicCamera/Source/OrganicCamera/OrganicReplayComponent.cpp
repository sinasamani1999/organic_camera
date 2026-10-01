#include "OrganicReplayComponent.h"

#include "OrganicCameraComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/PlayerController.h"
#include "HAL/FileManager.h"
#include "InputCoreTypes.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

UOrganicReplayComponent::UOrganicReplayComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	// Before physics: during playback the pawn is placed before the camera component (PostPhysics) reads it.
	PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

FString UOrganicReplayComponent::TakesDir()
{
	return FPaths::ProjectSavedDir() / TEXT("OrganicCamera") / TEXT("Takes");
}

void UOrganicReplayComponent::BeginPlay()
{
	Super::BeginPlay();
	OrganicCamera = GetOwner()->FindComponentByClass<UOrganicCameraComponent>();
	IFileManager::Get().MakeDirectory(*TakesDir(), true);
}

void UOrganicReplayComponent::EndPlay(const EEndPlayReason::Type Reason)
{
	if (State == EOrganicReplayState::Recording)
	{
		StopRecording();
	}
	else if (State == EOrganicReplayState::Playing)
	{
		StopPlayback();
	}
	Super::EndPlay(Reason);
}

void UOrganicReplayComponent::HandleKeys()
{
	APawn* Pawn = Cast<APawn>(GetOwner());
	APlayerController* PC = Pawn ? Cast<APlayerController>(Pawn->GetController()) : nullptr;
	if (!PC)
	{
		return;
	}
	if (PC->WasInputKeyJustPressed(EKeys::R))
	{
		if (State == EOrganicReplayState::Recording) StopRecording();
		else if (State == EOrganicReplayState::Idle) StartRecording();
	}
	if (PC->WasInputKeyJustPressed(EKeys::P))
	{
		if (State == EOrganicReplayState::Playing) StopPlayback();
		else if (State == EOrganicReplayState::Idle) StartPlayback();
	}
}

// ---------------------------------------------------------------- recording
void UOrganicReplayComponent::StartRecording()
{
	Samples.Reset();
	ClockTime = 0.0;
	CurrentTakeName = FString::Printf(TEXT("%s_%s.csv"), *TakePrefix, *FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S")));
	State = EOrganicReplayState::Recording;
	UE_LOG(LogTemp, Log, TEXT("OrganicReplay: recording %s (press R to stop)"), *CurrentTakeName);
}

void UOrganicReplayComponent::StopRecording()
{
	State = EOrganicReplayState::Idle;
	FString Csv = TEXT("time,x,y,z,pitch,yaw,roll,speed\n");
	for (const FOrganicReplaySample& S : Samples)
	{
		Csv += FString::Printf(TEXT("%.6f,%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%.4f\n"), S.Time,
			S.Location.X, S.Location.Y, S.Location.Z, S.Control.Pitch, S.Control.Yaw, S.Control.Roll, S.Speed);
	}
	const FString Path = TakesDir() / CurrentTakeName;
	FFileHelper::SaveStringToFile(Csv, *Path);
	UE_LOG(LogTemp, Log, TEXT("OrganicReplay: saved %d samples (%.1f s) to %s"), Samples.Num(), ClockTime, *Path);
}

void UOrganicReplayComponent::RecordTick(float DeltaTime)
{
	APawn* Pawn = Cast<APawn>(GetOwner());
	if (!Pawn)
	{
		return;
	}
	FOrganicReplaySample S;
	S.Time = ClockTime;
	S.Location = Pawn->GetActorLocation();
	S.Control = Pawn->GetControlRotation();
	S.Speed = Pawn->GetVelocity().Size2D() / 100.f;
	Samples.Add(S);
	ClockTime += DeltaTime;
}

// ---------------------------------------------------------------- playback
FString UOrganicReplayComponent::NewestTake() const
{
	TArray<FString> Files;
	IFileManager::Get().FindFiles(Files, *(TakesDir() / TEXT("*.csv")), true, false);
	FString Best;
	FDateTime BestTime = FDateTime::MinValue();
	for (const FString& F : Files)
	{
		const FDateTime T = IFileManager::Get().GetTimeStamp(*(TakesDir() / F));
		if (T > BestTime)
		{
			BestTime = T;
			Best = F;
		}
	}
	return Best;
}

bool UOrganicReplayComponent::LoadTake(const FString& Path)
{
	TArray<FString> Lines;
	if (!FFileHelper::LoadFileToStringArray(Lines, *Path) || Lines.Num() < 3)
	{
		return false;
	}
	Samples.Reset();
	for (int32 i = 1; i < Lines.Num(); ++i)  // skip header
	{
		TArray<FString> C;
		Lines[i].ParseIntoArray(C, TEXT(","));
		if (C.Num() < 8)
		{
			continue;
		}
		FOrganicReplaySample S;
		S.Time = FCString::Atod(*C[0]);
		S.Location = FVector(FCString::Atod(*C[1]), FCString::Atod(*C[2]), FCString::Atod(*C[3]));
		S.Control = FRotator(FCString::Atod(*C[4]), FCString::Atod(*C[5]), FCString::Atod(*C[6]));
		S.Speed = FCString::Atof(*C[7]);
		Samples.Add(S);
	}
	return Samples.Num() >= 2;
}

void UOrganicReplayComponent::SetPlayerInputEnabled(bool bEnabled)
{
	ACharacter* Character = Cast<ACharacter>(GetOwner());
	APlayerController* PC = Character ? Cast<APlayerController>(Character->GetController()) : nullptr;
	if (PC)
	{
		PC->SetIgnoreMoveInput(!bEnabled);
		PC->SetIgnoreLookInput(!bEnabled);
	}
	if (Character && Character->GetCharacterMovement())
	{
		if (bEnabled)
		{
			Character->GetCharacterMovement()->SetMovementMode(MOVE_Walking);
		}
		else
		{
			// The pawn is placed kinematically during playback; no physics may move it.
			Character->GetCharacterMovement()->StopMovementImmediately();
			Character->GetCharacterMovement()->DisableMovement();
		}
	}
}

bool UOrganicReplayComponent::StartPlayback()
{
	const FString Name = TakeToPlay.IsEmpty() ? NewestTake() : TakeToPlay;
	if (Name.IsEmpty() || !LoadTake(TakesDir() / Name))
	{
		UE_LOG(LogTemp, Warning, TEXT("OrganicReplay: no take to play in %s"), *TakesDir());
		return false;
	}
	CurrentTakeName = Name;
	ClockTime = 0.0;
	PlayIndex = 0;
	State = EOrganicReplayState::Playing;
	SetPlayerInputEnabled(false);

	// Same starting pose for every mode, and a fresh, deterministic camera layer.
	if (APawn* Pawn = Cast<APawn>(GetOwner()))
	{
		Pawn->SetActorLocation(Samples[0].Location, false, nullptr, ETeleportType::TeleportPhysics);
		if (AController* C = Pawn->GetController())
		{
			C->SetControlRotation(Samples[0].Control);
		}
	}
	if (OrganicCamera)
	{
		OrganicCamera->ResetLayer();
		OrganicCamera->SpeedOverride = Samples[0].Speed;
		if (bLogOffsetsDuringPlayback)
		{
			OrganicCamera->StartOffsetLog(FPaths::GetBaseFilename(Name));
		}
	}
	UE_LOG(LogTemp, Log, TEXT("OrganicReplay: playing %s (%.1f s)"), *Name, Samples.Last().Time);
	return true;
}

void UOrganicReplayComponent::StopPlayback()
{
	State = EOrganicReplayState::Idle;
	SetPlayerInputEnabled(true);
	if (OrganicCamera)
	{
		OrganicCamera->SpeedOverride = -1.f;
		OrganicCamera->StopOffsetLog();
	}
	UE_LOG(LogTemp, Log, TEXT("OrganicReplay: playback stopped at %.2f s"), ClockTime);
}

void UOrganicReplayComponent::PlayTick(float DeltaTime)
{
	ClockTime += DeltaTime;
	if (ClockTime >= Samples.Last().Time)
	{
		StopPlayback();
		return;
	}
	while (PlayIndex + 1 < Samples.Num() - 1 && Samples[PlayIndex + 1].Time <= ClockTime)
	{
		++PlayIndex;
	}
	const FOrganicReplaySample& A = Samples[PlayIndex];
	const FOrganicReplaySample& B = Samples[PlayIndex + 1];
	const float Alpha = static_cast<float>(FMath::Clamp((ClockTime - A.Time) / FMath::Max(B.Time - A.Time, 1e-6), 0.0, 1.0));

	const FVector Loc = FMath::Lerp(A.Location, B.Location, Alpha);
	const FRotator Rot = (A.Control + (B.Control - A.Control).GetNormalized() * Alpha).GetNormalized();

	APawn* Pawn = Cast<APawn>(GetOwner());
	if (!Pawn)
	{
		return;
	}
	Pawn->SetActorLocation(Loc, false, nullptr, ETeleportType::None);
	Pawn->SetActorRotation(FRotator(0.f, Rot.Yaw, 0.f));
	if (AController* C = Pawn->GetController())
	{
		C->SetControlRotation(Rot);
	}
	if (OrganicCamera)
	{
		OrganicCamera->SpeedOverride = FMath::Lerp(A.Speed, B.Speed, Alpha);
	}
}

void UOrganicReplayComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	HandleKeys();
	if (State == EOrganicReplayState::Recording)
	{
		RecordTick(DeltaTime);
	}
	else if (State == EOrganicReplayState::Playing)
	{
		PlayTick(DeltaTime);
	}
}
