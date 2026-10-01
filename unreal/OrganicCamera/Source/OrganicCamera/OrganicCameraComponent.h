// Organic camera layer: adds a small organic rotation on top of the player's control rotation.
// Modes: Off, Perlin (classic procedural noise), CameraShake (UE built-in), AI (residual from Python over OSC).

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "OSCMessage.h"
#include "OrganicCameraComponent.generated.h"

class UCameraComponent;
class UOSCClient;
class UOSCServer;
class UCameraShakeBase;

UENUM(BlueprintType)
enum class EOrganicCameraMode : uint8
{
	Off,
	Perlin,
	CameraShake,
	AI
};

UCLASS(ClassGroup = (Camera), meta = (BlueprintSpawnableComponent))
class ORGANICCAMERA_API UOrganicCameraComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UOrganicCameraComponent();

	/** Active layer. Keys 1-4 switch at runtime (1 Off, 2 Perlin, 3 CameraShake, 4 AI). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Organic Camera")
	EOrganicCameraMode Mode = EOrganicCameraMode::Off;

	/** Python listens here for /oc/state [frame, dt, intentYaw, intentPitch, speed]. */
	UPROPERTY(EditAnywhere, Category = "Organic Camera|OSC")
	int32 SendPort = 7001;

	/** UE listens here for /oc/residual [yaw, pitch] (degrees per frame). */
	UPROPERTY(EditAnywhere, Category = "Organic Camera|OSC")
	int32 ReceivePort = 7000;

	UPROPERTY(EditAnywhere, Category = "Organic Camera|Perlin")
	float PerlinAmplitudeDeg = 1.5f;

	UPROPERTY(EditAnywhere, Category = "Organic Camera|Perlin")
	float PerlinFrequencyHz = 1.2f;

	/** Assign a camera shake (e.g. a Blueprint using a Perlin/Wave pattern) for the CameraShake mode. */
	UPROPERTY(EditAnywhere, Category = "Organic Camera|CameraShake")
	TSubclassOf<UCameraShakeBase> ShakeClass;

	UPROPERTY(EditAnywhere, Category = "Organic Camera|CameraShake")
	float ShakeScale = 1.0f;

	/** Re-centring rate of the accumulated organic offset (1/s). Offset decays by exp(-LeakPerSecond * dt) each frame,
	 *  so the behaviour does not depend on the frame rate. 3.0/s equals the old 5% per frame at 60 fps. */
	UPROPERTY(EditAnywhere, Category = "Organic Camera", meta = (ClampMin = "0.0"))
	float LeakPerSecond = 3.0f;

	/** Duration of one model step. Each residual from Python is spread evenly over this time instead of
	 *  being applied in a single frame. */
	UPROPERTY(EditAnywhere, Category = "Organic Camera|AI", meta = (ClampMin = "0.001"))
	float ModelStepSeconds = 1.f / 30.f;

	UPROPERTY(EditAnywhere, Category = "Organic Camera")
	float MaxOffsetDeg = 3.0f;

	UFUNCTION(BlueprintCallable, Category = "Organic Camera")
	void SetMode(EOrganicCameraMode NewMode);

	/** Clears offset, noise time and pending residuals and restarts the shake (used at the start of a replay). */
	UFUNCTION(BlueprintCallable, Category = "Organic Camera")
	void ResetLayer();

	/** If >= 0, sent to Python instead of the pawn velocity (m/s). Set by the replay component during playback. */
	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Organic Camera")
	float SpeedOverride = -1.f;

	/** Per-frame CSV log of control rotation and camera offset (Saved/OrganicCamera/Logs). Key L toggles it. */
	UFUNCTION(BlueprintCallable, Category = "Organic Camera")
	void StartOffsetLog(const FString& Label);

	UFUNCTION(BlueprintCallable, Category = "Organic Camera")
	void StopOffsetLog();

	/** Frames since BeginPlay; sent to Python so logs from both sides can be aligned. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Organic Camera")
	int64 FrameIndex = 0;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
	UFUNCTION()
	void OnOscMessage(const FOSCMessage& Message, const FString& IPAddress, int32 Port);

	void HandleModeKeys();
	FRotator ComputeResidual(float DeltaTime);

	UPROPERTY()
	TObjectPtr<UCameraComponent> Camera;

	UPROPERTY()
	TObjectPtr<UOSCClient> OscClient;

	UPROPERTY()
	TObjectPtr<UOSCServer> OscServer;

	FRotator LastControlRotation = FRotator::ZeroRotator;
	FRotator Offset = FRotator::ZeroRotator;
	float Time = 0.f;

	FCriticalSection ResidualLock;
	FRotator LatestAIResidual = FRotator::ZeroRotator;  // newest residual not yet picked up by the game thread
	bool bNewAIResidual = false;
	double LastAIMessageTime = 0.0;

	// residual being spread over the current model step (game thread only)
	FRotator StepRate = FRotator::ZeroRotator;  // degrees per second
	float StepTimeLeft = 0.f;

	// offset log
	bool bLogging = false;
	FString LogLabel;
	double LogTime = 0.0;
	FRotator PrevControlForLog = FRotator::ZeroRotator;  // the camera manager's view is one frame old
	TArray<FString> LogLines;
};
