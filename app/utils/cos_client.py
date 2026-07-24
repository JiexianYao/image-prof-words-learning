"""
腾讯云COS操作客户端

提供对象存储服务的基本操作：
- 上传文件
- 下载文件
- 列出文件
- 删除文件
"""
from __future__ import annotations

import os
from typing import Optional, List, BinaryIO

from loguru import logger

from ..config import Settings, get_settings


class COSClient:
    """腾讯云COS客户端"""
    
    def __init__(self):
        """初始化COS客户端，从配置中读取密钥"""
        self._settings: Settings = get_settings()
        self._bucket_name = self._settings.cos_bucket_name
        self._region = self._settings.cos_region
        self._secret_id = self._settings.cos_secret_id
        self._secret_key = self._settings.cos_secret_key
        
        # 延迟导入，避免未安装cos_sdk时报错
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化COS客户端实例"""
        try:
            from qcloud_cos import CosConfig, CosS3Client
            
            config = CosConfig(
                Region=self._region,
                SecretId=self._secret_id,
                SecretKey=self._secret_key
            )
            self._client = CosS3Client(config)
            logger.info(f"COS客户端初始化成功，Bucket: {self._bucket_name}")
        except ImportError:
            logger.warning("cos_sdk未安装，请运行: pip install cos-python-sdk-v5")
        except Exception as e:
            logger.error(f"COS客户端初始化失败: {str(e)}")
    
    async def upload_file(
        self,
        file_obj: BinaryIO,
        object_key: str,
        content_type: Optional[str] = None
    ) -> bool:
        """
        上传文件到COS
        
        Args:
            file_obj: 文件对象（二进制模式）
            object_key: 对象键（文件路径）
            content_type: 内容类型（可选）
            
        Returns:
            是否上传成功
        """
        if not self._client:
            logger.error("COS客户端未初始化")
            return False
        
        try:
            kwargs = {
                'Bucket': self._bucket_name,
                'Key': object_key,
                'Body': file_obj
            }
            if content_type:
                kwargs['ContentType'] = content_type
            
            self._client.put_object(**kwargs)
            logger.info(f"文件上传成功: {object_key}")
            return True
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}")
            return False
    
    async def download_file(
        self,
        object_key: str,
        file_obj: Optional[BinaryIO] = None
    ) -> Optional[bytes]:
        """
        从COS下载文件
        
        Args:
            object_key: 对象键（文件路径）
            file_obj: 可选的文件对象，用于写入数据
            
        Returns:
            文件内容（如果未提供file_obj），否则返回None
        """
        if not self._client:
            logger.error("COS客户端未初始化")
            return None
        
        try:
            response = self._client.get_object(
                Bucket=self._bucket_name,
                Key=object_key
            )
            
            content = response['Body'].get_raw_stream().read()
            
            if file_obj:
                file_obj.write(content)
                logger.info(f"文件下载到对象: {object_key}")
            else:
                logger.info(f"文件下载成功: {object_key} ({len(content)} 字节)")
            
            return content
        except Exception as e:
            logger.error(f"文件下载失败: {str(e)}")
            return None
    
    async def list_files(
        self,
        prefix: str = "",
        max_keys: int = 100
    ) -> List[str]:
        """
        列出COS中的文件
        
        Args:
            prefix: 前缀过滤
            max_keys: 最大返回数量
            
        Returns:
            文件路径列表
        """
        if not self._client:
            logger.error("COS客户端未初始化")
            return []
        
        try:
            response = self._client.list_objects(
                Bucket=self._bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            contents = response.get('Contents', [])
            for obj in contents:
                files.append(obj['Key'])
            
            logger.info(f"列出文件: {len(files)} 个文件，前缀: {prefix}")
            return files
        except Exception as e:
            logger.error(f"列出文件失败: {str(e)}")
            return []
    
    async def delete_file(self, object_key: str) -> bool:
        """
        删除COS中的文件
        
        Args:
            object_key: 对象键（文件路径）
            
        Returns:
            是否删除成功
        """
        if not self._client:
            logger.error("COS客户端未初始化")
            return False
        
        try:
            self._client.delete_object(
                Bucket=self._bucket_name,
                Key=object_key
            )
            logger.info(f"文件删除成功: {object_key}")
            return True
        except Exception as e:
            logger.error(f"文件删除失败: {str(e)}")
            return False
    
    async def get_file_url(
        self,
        object_key: str,
        expires: int = 3600
    ) -> Optional[str]:
        """
        获取文件的预签名URL
        
        Args:
            object_key: 对象键（文件路径）
            expires: 过期时间（秒），默认1小时
            
        Returns:
            预签名URL
        """
        if not self._client:
            logger.error("COS客户端未初始化")
            return None
        
        try:
            url = self._client.get_presigned_url(
                Method='GET',
                Bucket=self._bucket_name,
                Key=object_key,
                Expired=expires
            )
            logger.info(f"获取文件URL: {object_key}")
            return url
        except Exception as e:
            logger.error(f"获取文件URL失败: {str(e)}")
            return None


# 注意：不在模块顶层实例化 COSClient()。那样只要 import 到本模块就会尝试连接 COS，
# 而 image 服务当前并未把 qcloud_cos 放进 requirements，会在 import 期抛错/告警。
# 需要用到时，由调用方显式 COSClient() 实例化即可。
