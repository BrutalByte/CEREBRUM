using UnrealBuildTool;
using System.Collections.Generic;

public class CerebrumVisualizerEditorTarget : TargetRules
{
	public CerebrumVisualizerEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		bOverrideBuildEnvironment = true;
		DefaultBuildSettings = BuildSettingsVersion.V5;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("CerebrumVisualizer");
	}
}
