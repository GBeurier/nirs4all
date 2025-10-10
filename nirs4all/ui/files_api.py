from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Dict, Any, List
import sys

from nirs4all.ui import utils
from typing import Optional
import subprocess
import shutil
from fastapi.concurrency import run_in_threadpool

router = APIRouter()


@router.get("/api/files/roots")
def files_roots() -> Dict[str, List[str]]:
    roots: List[str] = []
    if sys.platform.startswith("win"):
        from string import ascii_uppercase
        for letter in ascii_uppercase:
            candidate = Path(f"{letter}:/")
            if candidate.exists():
                roots.append(str(candidate))
    else:
        roots.append("/")
    return {"roots": roots}


@router.get("/api/files/list")
def files_list(path: str) -> Dict[str, Any]:
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="Path does not exist or is not a directory")
    return {"path": str(p), "items": utils.safe_list_dir(p)}


@router.get("/api/files/parent")
def files_parent(path: str) -> Dict[str, Any]:
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    parent = p.parent
    return {"parent": str(parent)}


def _pick_folder_dialog() -> Optional[str]:
    # Try tkinter first (cross-platform if available and running with display)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        res = filedialog.askdirectory()
        root.destroy()
        if res:
            return str(Path(res).resolve())
    except ImportError:
        # tkinter not available
        pass

    # Windows PowerShell fallback
    if sys.platform.startswith('win'):
        try:
            ps_cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Add-Type -AssemblyName System.Windows.Forms;"
                    "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
                    "if($d.ShowDialog() -eq 'OK'){ Write-Output $d.SelectedPath }"
                ),
            ]
            out = subprocess.run(ps_cmd, capture_output=True, text=True, check=False)
            path = out.stdout.strip()
            if path:
                return str(Path(path).resolve())
        except (subprocess.SubprocessError, OSError):
            pass

    # Linux: try zenity or kdialog
    if shutil.which('zenity'):
        try:
            out = subprocess.run(['zenity', '--file-selection', '--directory', '--title', 'Select folder'], capture_output=True, text=True, check=False)
            path = out.stdout.strip()
            if path:
                return str(Path(path).resolve())
        except (subprocess.SubprocessError, OSError):
            pass
    if shutil.which('kdialog'):
        try:
            out = subprocess.run(['kdialog', '--getexistingdirectory'], capture_output=True, text=True, check=False)
            path = out.stdout.strip()
            if path:
                return str(Path(path).resolve())
        except (subprocess.SubprocessError, OSError):
            pass

    return None


def _pick_files_dialog_tk(multiple: bool = True) -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        if multiple:
            res = filedialog.askopenfilenames()
        else:
            res = filedialog.askopenfilename()
        root.destroy()
        if not res:
            return []
        if isinstance(res, str):
            return [str(Path(res).resolve())]
        return [str(Path(p).resolve()) for p in list(res)]
    except (ImportError, RuntimeError, OSError):
        return []


def _pick_files_dialog_windows_ps(multiple: bool = True) -> List[str]:
    try:
        mult_flag = '$true' if multiple else '$false'
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                f"Add-Type -AssemblyName System.Windows.Forms; $ofd = New-Object System.Windows.Forms.OpenFileDialog; $ofd.Multiselect = {mult_flag};"
                " if($ofd.ShowDialog() -eq 'OK'){ $ofd.FileNames -join [Environment]::NewLine }"
            ),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        outp = out.stdout.strip()
        if outp:
            return [str(Path(p).resolve()) for p in outp.splitlines() if p.strip()]
    except (subprocess.SubprocessError, OSError):
        return []
    return []


def _pick_files_dialog_zenity(multiple: bool = True) -> List[str]:
    try:
        if not shutil.which('zenity'):
            return []
        cmd = ['zenity', '--file-selection']
        if multiple:
            cmd += ['--multiple', '--separator=|']
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        s = out.stdout.strip()
        if not s:
            return []
        parts = s.split('|') if multiple else [s]
        return [str(Path(p).resolve()) for p in parts if p]
    except (subprocess.SubprocessError, OSError):
        return []


def _pick_files_dialog_kdialog(multiple: bool = True) -> List[str]:
    try:
        if not shutil.which('kdialog'):
            return []
        if multiple:
            cmd = ['kdialog', '--getopenfilename', '--multiple', '--separate-output']
        else:
            cmd = ['kdialog', '--getopenfilename']
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        s = out.stdout.strip()
        if not s:
            return []
        parts = s.splitlines() if multiple else [s]
        return [str(Path(p).resolve()) for p in parts if p]
    except (subprocess.SubprocessError, OSError):
        return []


def _pick_files_dialog(multiple: bool = True) -> Optional[list]:
    # Try cross-platform Tkinter picker
    try:
        res = _pick_files_dialog_tk(multiple)
        if res:
            return res
    except (ImportError, RuntimeError, OSError):
        pass

    # Windows PowerShell helper
    if sys.platform.startswith('win'):
        try:
            res = _pick_files_dialog_windows_ps(multiple)
            if res:
                return res
        except (subprocess.SubprocessError, OSError):
            pass

    # Try zenity
    try:
        res = _pick_files_dialog_zenity(multiple)
        if res:
            return res
    except (subprocess.SubprocessError, OSError):
        pass

    # Try kdialog
    try:
        res = _pick_files_dialog_kdialog(multiple)
        if res:
            return res
    except (subprocess.SubprocessError, OSError):
        pass

    return []


@router.post('/api/files/dialog/folder')
async def files_dialog_folder() -> Dict[str, Any]:
    try:
        path = await run_in_threadpool(_pick_folder_dialog)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open folder dialog: {e}") from e
    if not path:
        return {"path": None}
    return {"path": str(path)}


@router.post('/api/files/dialog/files')
async def files_dialog_files(multiple: bool = True) -> Dict[str, Any]:
    try:
        paths = await run_in_threadpool(_pick_files_dialog, multiple)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open files dialog: {e}") from e
    return {"paths": paths}
