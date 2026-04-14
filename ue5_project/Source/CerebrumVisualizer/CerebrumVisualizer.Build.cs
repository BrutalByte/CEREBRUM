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
            "WebsocketBlueprint",
            "Json",
            "JsonUtilities"
        });
	}
}
