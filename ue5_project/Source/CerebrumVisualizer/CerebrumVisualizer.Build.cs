using UnrealBuildTool;

public class CerebrumVisualizer : ModuleRules
{
	public CerebrumVisualizer(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
	
		PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "EasyWebsockets",
            "Json",
            "JsonUtilities",
            "HTTP",         // REST graph pre-load in CerebrumBrain
            "UMG",          // UCerebrumQueryWidget / UCerebrumHUDOverlay
            "Slate",
            "SlateCore",
        });
	}
}
