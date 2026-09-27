// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class OrganicCamera : ModuleRules
{
	public OrganicCamera(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "EnhancedInput", "OSC" });
	}
}
