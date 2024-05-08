import paramiko

from ..typing import (
    Path,
    Optional,
    Transport,
    SFTPClient,
)

__all__ = [
    "RemoteConnector",
]


class RemoteConnector:
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

    @staticmethod
    def init_sftp(
        transport: Transport,
    ) -> SFTPClient:
        sftp = paramiko.SFTPClient.from_transport(transport)

        return sftp

    @staticmethod
    def close(
        transport: Transport,
        sftp: SFTPClient,
    ) -> None:
        transport.close()
        sftp.close()

    def download(
        self,
        remote_path: Path,
        local_path: Path,
    ) -> None:
        transport = self.init_transport()

        sftp = self.init_sftp(transport)
        sftp.get(remote_path, local_path)

        self.close(transport, sftp)

    def upload(
        self,
        local_path: Path,
        remote_path: Path,
    ) -> None:
        transport = self.init_transport()

        sftp = self.init_sftp(transport)
        sftp.put(local_path, remote_path)

        self.close(transport, sftp)
