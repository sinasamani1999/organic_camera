// Records the player's path (location + control rotation) and plays it back exactly,
// so the same movement can be filmed with each camera mode (Off / Perlin / CameraShake / AI).
//
// Keys at runtime:  R = start/stop recording,  P = play/stop the latest (or named) take.
// Takes are saved as CSV in <Project>/Saved/OrganicCamera/Takes/.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "OrganicReplayComponent.generated.h"

class UOrganicCameraComponent;

struct FOrganicReplaySample
{
	double Time = 0.0;
	FVector Location = FVector::ZeroVector;
	FRotator Control = FRotator::ZeroRotator;
	float Speed = 0.f;  // m/s, horizontal
};

UENUM(BlueprintType)
enum class EOrganicReplayState : uint8
{
	Idle,
	Recording,
	Playing
};

UCLASS(ClassGroup = (Camera), meta = (BlueprintSpawnableComponent))
class ORGANICCAMERA_API UOrganicReplayComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UOrganicReplayComponent();

	/** Take to play (file name without folder, e.g. "take_corridor_01.csv"). Empty = newest take. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Organic Replay")
	FString TakeToPlay;

	/** Name prefix for new recordings. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Organic Replay")
	FString TakePrefix = TEXT("take");

	/** Write a camera-offset log (for RMS matching) while a take is playing. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Organic Replay")
	bool bLogOffsetsDuringPlayback = true;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Organic Replay")
	EOrganicReplayState State = EOrganicReplayState::Idle;

	UFUNCTION(BlueprintCallable, Category = "Organic Replay")
	void StartRecording();

	UFUNCTION(BlueprintCallable, Category = "Organic Replay")
	void StopRecording();

	UFUNCTION(BlueprintCallable, Category = "Organic Replay")
	bool StartPlayback();

	UFUNCTION(BlueprintCallable, Category = "Organic Replay")
	void StopPlayback();

	static FString TakesDir();

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
	void HandleKeys();
	void RecordTick(float DeltaTime);
	void PlayTick(float DeltaTime);
	bool LoadTake(const FString& Path);
	FString NewestTake() const;
	void SetPlayerInputEnabled(bool bEnabled);

	UPROPERTY()
	TObjectPtr<UOrganicCameraComponent> OrganicCamera;

	TArray<FOrganicReplaySample> Samples;
	double ClockTime = 0.0;   // seconds since record/playback start
	int32 PlayIndex = 0;
	FString CurrentTakeName;
};
