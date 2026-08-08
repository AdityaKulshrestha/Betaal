; Inno Setup script for Betaal.
;
; Compile with:  iscc installer\betaal.iss   (run scripts\build.ps1 -Installer)
; Expects the PyInstaller bundle at ..\dist\Betaal\ (produced by betaal.spec).
;
; Per-user install (no admin required) so the post-install model download/
; optimization writes into the installing user's ~/.cache/betaal, and the
; global hotkey runs in that user's session.

#define AppName "Betaal"
#define AppVersion "0.1.0"
#define AppPublisher "Betaal"
#define AppExeName "Betaal.exe"

[Setup]
AppId={{C4E1F0B2-6A3D-4C9E-9F2A-7B1E0D5C8A10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=Output
OutputBaseFilename=Betaal-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\assets\betaal.ico
UninstallDisplayIcon={app}\{#AppExeName}
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "startupicon"; Description: "Start {#AppName} automatically at login (runs in the background tray)"; GroupDescription: "Startup:"
Name: "prepareModels"; Description: "Download and optimize models now (recommended; requires internet)"; GroupDescription: "First-run setup:"

[Files]
; Copy the entire PyInstaller one-directory bundle.
Source: "..\dist\Betaal\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: startupicon
; Launch at login -> the app starts minimized to the system tray (background).
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: startupicon

[Run]
; Download + device-specific optimization on the target machine (user context).
Filename: "{app}\{#AppExeName}"; Parameters: "setup"; Description: "Downloading and optimizing models"; StatusMsg: "Downloading and optimizing models (this can take a while)..."; Flags: runhidden waituntilterminated; Tasks: prepareModels
; Optionally launch the app right after install.
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
