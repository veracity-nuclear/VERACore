
[Setup]
AppName=VeraCore
AppVersion=1.3.1
DefaultDirName={autopf}\VeraCore
DefaultGroupName=VeraCore
OutputBaseFilename=VeraCoreSetup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin

[Files]
Source: "dist\VeraCore\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs
; The Evergreen bootstrapper, downloaded from Microsoft and placed next to this script
Source: "MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\VeraCore"; Filename: "{app}\VeraCore.exe"
Name: "{commondesktop}\VeraCore"; Filename: "{app}\VeraCore.exe"

[Run]
; Install WebView2 only if it isn't already present; bootstrapper is a no-op if it is
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; \
  Check: WebView2Missing; StatusMsg: "Installing WebView2 Runtime..."
Filename: "{app}\VeraCore.exe"; Description: "Launch VeraCore"; \
  Flags: nowait postinstall skipifsilent

[Code]
function WebView2Missing: Boolean;
var
  Version: String;
begin
  // WebView2 records its version under this key (per-machine, 64-bit view) when installed.
  // If we can't read a version, treat it as missing and run the bootstrapper.
  Result := not RegQueryStringValue(HKLM,
    'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
    'pv', Version);
end;

// Remove all *.dist-info directories left over from a previous installation so
// that importlib.metadata always finds exactly one version of each package.
procedure RemoveStaleDistInfo(InternalDir: String);
var
  FindRec: TFindRec;
  DirsToDelete: TStringList;
  I: Integer;
begin
  DirsToDelete := TStringList.Create;
  try
    if FindFirst(InternalDir + '\*.dist-info', FindRec) then begin
      try
        repeat
          if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
            DirsToDelete.Add(InternalDir + '\' + FindRec.Name);
        until not FindNext(FindRec);
      finally
        FindClose(FindRec);
      end;
    end;
    for I := 0 to DirsToDelete.Count - 1 do
      DelTree(DirsToDelete.Strings[I], True, True, True);
  finally
    DirsToDelete.Free;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
    RemoveStaleDistInfo(ExpandConstant('{app}\_internal'));
end;