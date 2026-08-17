#!/usr/bin/env python3
"""Generate EfficientSAMDemo.xcodeproj and copy the Core ML models in.

Run from this directory:

    python3 generate_project.py

Writing project.pbxproj by hand is unpleasant but avoids requiring xcodegen or
tuist just to open a three-file app.
"""
import os
import plistlib
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
APP = "EfficientSAMDemo"
SRC_DIR = os.path.join(HERE, APP)
PROJ_DIR = os.path.join(HERE, f"{APP}.xcodeproj")

SOURCES = ["EfficientSAMDemoApp.swift", "ContentView.swift", "Segmenter.swift"]
MODELS = [
    "efficient_sam_vitt_encoder.mlpackage",
    "efficient_sam_vitt_decoder.mlpackage",
]

# Deterministic 24-hex-char object IDs. Xcode only requires uniqueness.
def oid(n):
    return f"{n:024X}"

IDS = {}
_counter = [0x1000]

def ref(key):
    if key not in IDS:
        _counter[0] += 1
        IDS[key] = oid(_counter[0])
    return IDS[key]


def copy_models():
    """Copy the exported .mlpackage bundles next to the sources."""
    found = []
    for m in MODELS:
        src = os.path.join(REPO, "weights", "coreml", m)
        dst = os.path.join(SRC_DIR, m)
        if not os.path.isdir(src):
            print(f"  !! missing {src}")
            print("     Run notebooks/EfficientSAM_coreml_export.ipynb first.")
            continue
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  copied {m}")
        found.append(m)
    return found


def build_pbxproj(models):
    resources = list(models)

    file_refs, build_files = [], []
    sources_phase, resources_phase = [], []

    for name in SOURCES:
        fr, bf = ref(f"fr:{name}"), ref(f"bf:{name}")
        file_refs.append(
            f'\t\t{fr} /* {name} */ = {{isa = PBXFileReference; '
            f'lastKnownFileType = sourcecode.swift; path = "{name}"; '
            f"sourceTree = \"<group>\"; }};"
        )
        build_files.append(
            f"\t\t{bf} /* {name} in Sources */ = {{isa = PBXBuildFile; "
            f"fileRef = {fr} /* {name} */; }};"
        )
        sources_phase.append(f"\t\t\t\t{bf} /* {name} in Sources */,")

    for name in resources:
        fr, bf = ref(f"fr:{name}"), ref(f"bf:{name}")
        file_refs.append(
            f'\t\t{fr} /* {name} */ = {{isa = PBXFileReference; '
            f'lastKnownFileType = folder.mlpackage; path = "{name}"; '
            f"sourceTree = \"<group>\"; }};"
        )
        build_files.append(
            f"\t\t{bf} /* {name} in Resources */ = {{isa = PBXBuildFile; "
            f"fileRef = {fr} /* {name} */; }};"
        )
        resources_phase.append(f"\t\t\t\t{bf} /* {name} in Resources */,")

    group_children = "\n".join(
        f"\t\t\t\t{ref('fr:' + n)} /* {n} */," for n in SOURCES + resources
    )
    info_fr = ref("fr:Info.plist")
    file_refs.append(
        f"\t\t{info_fr} /* Info.plist */ = {{isa = PBXFileReference; "
        f'lastKnownFileType = text.plist.xml; path = "Info.plist"; '
        f"sourceTree = \"<group>\"; }};"
    )
    group_children += f"\n\t\t\t\t{info_fr} /* Info.plist */,"

    P = {k: ref(k) for k in [
        "project", "target", "product", "mainGroup", "appGroup", "productsGroup",
        "sourcesPhase", "frameworksPhase", "resourcesPhase",
        "projConfigList", "targetConfigList",
        "projDebug", "projRelease", "targetDebug", "targetRelease",
    ]}

    common_target = """
\t\t\t\tASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;
\t\t\t\tCODE_SIGN_STYLE = Automatic;
\t\t\t\tCURRENT_PROJECT_VERSION = 1;
\t\t\t\tDEVELOPMENT_TEAM = "";
\t\t\t\tENABLE_PREVIEWS = YES;
\t\t\t\tGENERATE_INFOPLIST_FILE = NO;
\t\t\t\tINFOPLIST_FILE = "EfficientSAMDemo/Info.plist";
\t\t\t\tIPHONEOS_DEPLOYMENT_TARGET = 17.0;
\t\t\t\tLD_RUNPATH_SEARCH_PATHS = (
\t\t\t\t\t"$(inherited)",
\t\t\t\t\t"@executable_path/Frameworks",
\t\t\t\t);
\t\t\t\tMARKETING_VERSION = 1.0;
\t\t\t\tPRODUCT_BUNDLE_IDENTIFIER = "com.example.EfficientSAMDemo";
\t\t\t\tPRODUCT_NAME = "$(TARGET_NAME)";
\t\t\t\tSWIFT_EMIT_LOC_STRINGS = YES;
\t\t\t\tSWIFT_VERSION = 5.0;
\t\t\t\tTARGETED_DEVICE_FAMILY = "1,2";
"""

    common_proj = """
\t\t\t\tALWAYS_SEARCH_USER_PATHS = NO;
\t\t\t\tCLANG_ENABLE_MODULES = YES;
\t\t\t\tCLANG_ENABLE_OBJC_ARC = YES;
\t\t\t\tCOPY_PHASE_STRIP = NO;
\t\t\t\tENABLE_STRICT_OBJC_MSGSEND = YES;
\t\t\t\tGCC_C_LANGUAGE_STANDARD = gnu17;
\t\t\t\tIPHONEOS_DEPLOYMENT_TARGET = 17.0;
\t\t\t\tSDKROOT = iphoneos;
\t\t\t\tSWIFT_VERSION = 5.0;
"""

    return f"""// !$*UTF8*$!
{{
\tarchiveVersion = 1;
\tclasses = {{
\t}};
\tobjectVersion = 56;
\tobjects = {{

/* Begin PBXBuildFile section */
{chr(10).join(build_files)}
/* End PBXBuildFile section */

/* Begin PBXFileReference section */
\t\t{P['product']} /* {APP}.app */ = {{isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = "{APP}.app"; sourceTree = BUILT_PRODUCTS_DIR; }};
{chr(10).join(file_refs)}
/* End PBXFileReference section */

/* Begin PBXFrameworksBuildPhase section */
\t\t{P['frameworksPhase']} /* Frameworks */ = {{
\t\t\tisa = PBXFrameworksBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXFrameworksBuildPhase section */

/* Begin PBXGroup section */
\t\t{P['mainGroup']} = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
\t\t\t\t{P['appGroup']} /* {APP} */,
\t\t\t\t{P['productsGroup']} /* Products */,
\t\t\t);
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{P['appGroup']} /* {APP} */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
{group_children}
\t\t\t);
\t\t\tpath = "{APP}";
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{P['productsGroup']} /* Products */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
\t\t\t\t{P['product']} /* {APP}.app */,
\t\t\t);
\t\t\tname = Products;
\t\t\tsourceTree = "<group>";
\t\t}};
/* End PBXGroup section */

/* Begin PBXNativeTarget section */
\t\t{P['target']} /* {APP} */ = {{
\t\t\tisa = PBXNativeTarget;
\t\t\tbuildConfigurationList = {P['targetConfigList']} /* Build configuration list for PBXNativeTarget "{APP}" */;
\t\t\tbuildPhases = (
\t\t\t\t{P['sourcesPhase']} /* Sources */,
\t\t\t\t{P['frameworksPhase']} /* Frameworks */,
\t\t\t\t{P['resourcesPhase']} /* Resources */,
\t\t\t);
\t\t\tbuildRules = (
\t\t\t);
\t\t\tdependencies = (
\t\t\t);
\t\t\tname = "{APP}";
\t\t\tproductName = "{APP}";
\t\t\tproductReference = {P['product']} /* {APP}.app */;
\t\t\tproductType = "com.apple.product-type.application";
\t\t}};
/* End PBXNativeTarget section */

/* Begin PBXProject section */
\t\t{P['project']} /* Project object */ = {{
\t\t\tisa = PBXProject;
\t\t\tattributes = {{
\t\t\t\tBuildIndependentTargetsInParallel = 1;
\t\t\t\tLastSwiftUpdateCheck = 1500;
\t\t\t\tLastUpgradeCheck = 1500;
\t\t\t\tTargetAttributes = {{
\t\t\t\t\t{P['target']} = {{
\t\t\t\t\t\tCreatedOnToolsVersion = 15.0;
\t\t\t\t\t}};
\t\t\t\t}};
\t\t\t}};
\t\t\tbuildConfigurationList = {P['projConfigList']} /* Build configuration list for PBXProject "{APP}" */;
\t\t\tcompatibilityVersion = "Xcode 14.0";
\t\t\tdevelopmentRegion = en;
\t\t\thasScannedForEncodings = 0;
\t\t\tknownRegions = (
\t\t\t\ten,
\t\t\t\tBase,
\t\t\t);
\t\t\tmainGroup = {P['mainGroup']};
\t\t\tproductRefGroup = {P['productsGroup']} /* Products */;
\t\t\tprojectDirPath = "";
\t\t\tprojectRoot = "";
\t\t\ttargets = (
\t\t\t\t{P['target']} /* {APP} */,
\t\t\t);
\t\t}};
/* End PBXProject section */

/* Begin PBXResourcesBuildPhase section */
\t\t{P['resourcesPhase']} /* Resources */ = {{
\t\t\tisa = PBXResourcesBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
{chr(10).join(resources_phase)}
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXResourcesBuildPhase section */

/* Begin PBXSourcesBuildPhase section */
\t\t{P['sourcesPhase']} /* Sources */ = {{
\t\t\tisa = PBXSourcesBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
{chr(10).join(sources_phase)}
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXSourcesBuildPhase section */

/* Begin XCBuildConfiguration section */
\t\t{P['projDebug']} /* Debug */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{{common_proj}\t\t\t\tDEBUG_INFORMATION_FORMAT = dwarf;
\t\t\t\tENABLE_TESTABILITY = YES;
\t\t\t\tGCC_OPTIMIZATION_LEVEL = 0;
\t\t\t\tMTL_ENABLE_DEBUG_INFO = INCLUDE_SOURCE;
\t\t\t\tONLY_ACTIVE_ARCH = YES;
\t\t\t\tSWIFT_ACTIVE_COMPILATION_CONDITIONS = "DEBUG $(inherited)";
\t\t\t\tSWIFT_OPTIMIZATION_LEVEL = "-Onone";
\t\t\t}};
\t\t\tname = Debug;
\t\t}};
\t\t{P['projRelease']} /* Release */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{{common_proj}\t\t\t\tDEBUG_INFORMATION_FORMAT = "dwarf-with-dsym";
\t\t\t\tENABLE_NS_ASSERTIONS = NO;
\t\t\t\tMTL_ENABLE_DEBUG_INFO = NO;
\t\t\t\tSWIFT_COMPILATION_MODE = wholemodule;
\t\t\t}};
\t\t\tname = Release;
\t\t}};
\t\t{P['targetDebug']} /* Debug */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{{common_target}\t\t\t}};
\t\t\tname = Debug;
\t\t}};
\t\t{P['targetRelease']} /* Release */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{{common_target}\t\t\t}};
\t\t\tname = Release;
\t\t}};
/* End XCBuildConfiguration section */

/* Begin XCConfigurationList section */
\t\t{P['projConfigList']} /* Build configuration list for PBXProject "{APP}" */ = {{
\t\t\tisa = XCConfigurationList;
\t\t\tbuildConfigurations = (
\t\t\t\t{P['projDebug']} /* Debug */,
\t\t\t\t{P['projRelease']} /* Release */,
\t\t\t);
\t\t\tdefaultConfigurationIsVisible = 0;
\t\t\tdefaultConfigurationName = Release;
\t\t}};
\t\t{P['targetConfigList']} /* Build configuration list for PBXNativeTarget "{APP}" */ = {{
\t\t\tisa = XCConfigurationList;
\t\t\tbuildConfigurations = (
\t\t\t\t{P['targetDebug']} /* Debug */,
\t\t\t\t{P['targetRelease']} /* Release */,
\t\t\t);
\t\t\tdefaultConfigurationIsVisible = 0;
\t\t\tdefaultConfigurationName = Release;
\t\t}};
/* End XCConfigurationList section */
\t}};
\trootObject = {P['project']} /* Project object */;
}}
"""


def write_info_plist():
    info = {
        "CFBundleDevelopmentRegion": "$(DEVELOPMENT_LANGUAGE)",
        "CFBundleExecutable": "$(EXECUTABLE_NAME)",
        "CFBundleIdentifier": "$(PRODUCT_BUNDLE_IDENTIFIER)",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "$(PRODUCT_NAME)",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "$(MARKETING_VERSION)",
        "CFBundleVersion": "$(CURRENT_PROJECT_VERSION)",
        "LSRequiresIPhoneOS": True,
        # PhotosPicker runs out of process, so no photo-library permission is
        # required. The key is here only for the fallback path.
        "NSPhotoLibraryUsageDescription":
            "Pick a photo to segment with EfficientSAM.",
        "UILaunchScreen": {},
        "UISupportedInterfaceOrientations": [
            "UIInterfaceOrientationPortrait",
            "UIInterfaceOrientationLandscapeLeft",
            "UIInterfaceOrientationLandscapeRight",
        ],
    }
    path = os.path.join(SRC_DIR, "Info.plist")
    with open(path, "wb") as f:
        plistlib.dump(info, f)
    print("  wrote Info.plist")


def write_scheme():
    scheme_dir = os.path.join(PROJ_DIR, "xcshareddata", "xcschemes")
    os.makedirs(scheme_dir, exist_ok=True)
    target_id, product_id = ref("target"), ref("product")
    scheme = f"""<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="1500" version="1.7">
   <BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES">
      <BuildActionEntries>
         <BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">
            <BuildableReference
               BuildableIdentifier="primary"
               BlueprintIdentifier="{target_id}"
               BuildableName="{APP}.app"
               BlueprintName="{APP}"
               ReferencedContainer="container:{APP}.xcodeproj">
            </BuildableReference>
         </BuildActionEntry>
      </BuildActionEntries>
   </BuildAction>
   <LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.DebuggerFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES">
      <BuildableProductRunnable runnableDebuggingMode="0">
         <BuildableReference
            BuildableIdentifier="primary"
            BlueprintIdentifier="{target_id}"
            BuildableName="{APP}.app"
            BlueprintName="{APP}"
            ReferencedContainer="container:{APP}.xcodeproj">
         </BuildableReference>
      </BuildableProductRunnable>
   </LaunchAction>
   <ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES">
      <BuildableProductRunnable runnableDebuggingMode="0">
         <BuildableReference
            BuildableIdentifier="primary"
            BlueprintIdentifier="{target_id}"
            BuildableName="{APP}.app"
            BlueprintName="{APP}"
            ReferencedContainer="container:{APP}.xcodeproj">
         </BuildableReference>
      </BuildableProductRunnable>
   </ProfileAction>
   <AnalyzeAction buildConfiguration="Debug"></AnalyzeAction>
   <ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"></ArchiveAction>
</Scheme>
"""
    with open(os.path.join(scheme_dir, f"{APP}.xcscheme"), "w") as f:
        f.write(scheme)
    print("  wrote shared scheme")


def main():
    print("Copying Core ML models…")
    models = copy_models()

    print("Generating project…")
    os.makedirs(PROJ_DIR, exist_ok=True)
    write_info_plist()
    with open(os.path.join(PROJ_DIR, "project.pbxproj"), "w") as f:
        f.write(build_pbxproj(models))
    print("  wrote project.pbxproj")
    write_scheme()

    if not models:
        print("\nWARNING: no models bundled — the app will show a load error.")

    print(f"\nDone. Open with:\n  open {os.path.relpath(PROJ_DIR, os.getcwd())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
