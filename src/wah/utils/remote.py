import paramiko

from ..typing import (
    Path,
    Optional,
    Transport,
)

__all__ = [
    "RemoteDownloader",
]


class RemoteDownloader:
    def __init__(
        self,
        host: str,
        port: int,
        username: str = "",
        password: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def init_transport(self) -> Transport:
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)

        return transport

    def download(
        self,
        remote_path: Path,
        local_path: Path,
    ) -> None:
        transport: Transport = self.init_transport()

        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get(remote_path, local_path)

        sftp.close()
        transport.close()
