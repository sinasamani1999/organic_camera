// Copyright Epic Games, Inc. All Rights Reserved.

#include "OrganicCameraGameMode.h"
#include "OrganicCameraCharacter.h"
#include "UObject/ConstructorHelpers.h"

AOrganicCameraGameMode::AOrganicCameraGameMode()
	: Super()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnClassFinder(TEXT("/Game/FirstPerson/Blueprints/BP_FirstPersonCharacter"));
	DefaultPawnClass = PlayerPawnClassFinder.Class;

}
