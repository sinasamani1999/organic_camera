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

	/** Fraction of the accumulated organic offset removed per frame, keeps the offset centred. */
	UPROPERTY(EditAnywhere, Category = "Organic Camera", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float Leak = 0.05f;

	UPROPERTY(EditAnywhere, Category = "Organic Camera")
	float MaxOffsetDeg = 3.0f;

	UFUNCTION(BlueprintCallable, Category = "Organic Camera")
	void SetMode(EOrganicCameraMode NewMode);

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
	FRotator LatestAIResidual = FRotator::ZeroRotator;
	double LastAIMessageTime = 0.0;
};
