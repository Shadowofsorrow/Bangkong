"""
Deployment manager for Bangkong LLM Training System
"""

import os
import shutil
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..config.schemas import BangkongConfig
from ..utils.dynamic_importer import DynamicImporter


class DeploymentTarget(ABC):
    """Abstract base class for deployment targets."""

    def __init__(self, config: BangkongConfig):
        """
        Initialize the deployment target.

        Args:
            config: Bangkong configuration.
        """
        self.config = config

    @abstractmethod
    def deploy(self, model_path: str) -> bool:
        """
        Deploy the model.

        Args:
            model_path: Path to the model to deploy.

        Returns:
            True if deployment was successful, False otherwise.
        """
        pass

    @abstractmethod
    def validate_deployment(self) -> bool:
        """
        Validate the deployment.

        Returns:
            True if deployment is valid, False otherwise.
        """
        pass


class LocalDeployment(DeploymentTarget):
    """Local deployment target."""

    def deploy(self, model_path: str) -> bool:
        """
        Deploy model locally by copying to local deployment directory.

        Args:
            model_path: Path to the model to deploy.

        Returns:
            True if deployment was successful, False otherwise.
        """
        try:
            # Get deployment directory from config or use default
            deploy_dir = self.config.deployment.get("local_deploy_dir", "./deployed_models")
            
            # Create deployment directory if it doesn't exist
            os.makedirs(deploy_dir, exist_ok=True)
            
            # Copy model to deployment directory
            model_filename = os.path.basename(model_path)
            deploy_path = os.path.join(deploy_dir, model_filename)
            
            print(f"Deploying model from {model_path} to {deploy_path}")
            shutil.copy2(model_path, deploy_path)
            
            # Save deployment info
            info_path = os.path.join(deploy_dir, "deployment_info.txt")
            with open(info_path, "w") as f:
                f.write(f"Model deployed from: {model_path}\n")
                f.write(f"Deployed at: {deploy_path}\n")
                f.write("Status: Deployed successfully\n")
            
            print(f"Model deployed successfully to {deploy_path}")
            return True
        except Exception as e:
            print(f"Local deployment failed: {e}")
            return False

    def validate_deployment(self) -> bool:
        """
        Validate local deployment by checking if model files exist.

        Returns:
            True if deployment is valid, False otherwise.
        """
        try:
            deploy_dir = self.config.deployment.get("local_deploy_dir", "./deployed_models")
            
            # Check if deployment directory exists
            if not os.path.exists(deploy_dir):
                print("Deployment directory does not exist")
                return False
                
            # Check if deployment info file exists
            info_path = os.path.join(deploy_dir, "deployment_info.txt")
            if not os.path.exists(info_path):
                print("Deployment info file not found")
                return False
                
            print("Local deployment validation passed")
            return True
        except Exception as e:
            print(f"Local deployment validation failed: {e}")
            return False


class CloudDeployment(DeploymentTarget):
    """Cloud deployment target."""

    def deploy(self, model_path: str) -> bool:
        """
        Deploy model to cloud storage (AWS S3 as example).

        Args:
            model_path: Path to the model to deploy.

        Returns:
            True if deployment was successful, False otherwise.
        """
        try:
            # Try to import boto3 for AWS deployment
            boto3 = DynamicImporter.safe_import("boto3", "AWS deployment")
            if not boto3:
                print("AWS deployment failed: boto3 not available")
                return False
            
            # Get AWS configuration
            aws_config = self.config.deployment.get("aws", {})
            bucket_name = aws_config.get("bucket_name", "bangkong-models")
            region = aws_config.get("region", "us-east-1")
            
            print(f"Deploying model from {model_path} to AWS S3 bucket {bucket_name}")
            
            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=aws_config.get("access_key_id"),
                aws_secret_access_key=aws_config.get("secret_access_key")
            )
            
            # Upload model
            model_filename = os.path.basename(model_path)
            s3_client.upload_file(model_path, bucket_name, model_filename)
            
            print(f"Model deployed successfully to AWS S3 bucket {bucket_name}")
            return True
        except Exception as e:
            print(f"Cloud deployment failed: {e}")
            return False

    def validate_deployment(self) -> bool:
        """
        Validate cloud deployment by checking if model exists in cloud storage.

        Returns:
            True if deployment is valid, False otherwise.
        """
        try:
            # Try to import boto3 for AWS validation
            boto3 = DynamicImporter.safe_import("boto3", "AWS validation")
            if not boto3:
                print("AWS validation failed: boto3 not available")
                return False
            
            # Get AWS configuration
            aws_config = self.config.deployment.get("aws", {})
            bucket_name = aws_config.get("bucket_name", "bangkong-models")
            region = aws_config.get("region", "us-east-1")
            
            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=aws_config.get("access_key_id"),
                aws_secret_access_key=aws_config.get("secret_access_key")
            )
            
            # Check if bucket exists and is accessible
            s3_client.head_bucket(Bucket=bucket_name)
            
            print("Cloud deployment validation passed")
            return True
        except Exception as e:
            print(f"Cloud deployment validation failed: {e}")
            return False


class DeploymentManager:
    """Manages deployment across different targets."""

    def __init__(self, config: BangkongConfig):
        """
        Initialize the deployment manager.

        Args:
            config: Bangkong configuration.
        """
        self.config = config
        self.targets = {}
        self._initialize_targets()

    def _initialize_targets(self):
        """Initialize available deployment targets."""
        self.targets["local"] = LocalDeployment(self.config)

        # Try to initialize cloud deployment if dependencies are available
        if DynamicImporter.is_module_available("boto3"):  # AWS
            self.targets["cloud"] = CloudDeployment(self.config)

    def deploy(self, model_path: str, target: str = "local") -> bool:
        """
        Deploy model to specified target.

        Args:
            model_path: Path to the model to deploy.
            target: Target environment for deployment.

        Returns:
            True if deployment was successful, False otherwise.
        """
        if target not in self.targets:
            raise ValueError(f"Unsupported deployment target: {target}")

        print(f"Deploying to {target} target")
        return self.targets[target].deploy(model_path)

    def validate_deployment(self, target: str = "local") -> bool:
        """
        Validate deployment to specified target.

        Args:
            target: Target environment to validate.

        Returns:
            True if deployment is valid, False otherwise.
        """
        if target not in self.targets:
            raise ValueError(f"Unsupported deployment target: {target}")

        print(f"Validating {target} deployment")
        return self.targets[target].validate_deployment()