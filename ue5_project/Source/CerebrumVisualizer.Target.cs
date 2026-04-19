using UnrealBuildTool;
using System.Collections.Generic;

public class CerebrumVisualizerTarget : TargetRules
{
	public CerebrumVisualizerTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		bOverrideBuildEnvironment = true;
		DefaultBuildSettings = BuildSettingsVersion.V5;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("CerebrumVisualizer");
	}
}
