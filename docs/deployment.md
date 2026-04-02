# Bangkong Deployment Guide

## Overview

The Bangkong LLM Training System supports multiple deployment scenarios, from local development to cloud production environments. This guide explains how to deploy models trained with Bangkong.

## Deployment Targets

The system supports three main deployment targets:

1. **Local**: Direct deployment on the local machine
2. **Cloud**: Deployment to cloud platforms (AWS, GCP, Azure)
3. **Hybrid**: Combination of local and cloud deployment

## Local Deployment

### Prerequisites

- Python 3.8+
- Required dependencies installed
- Trained model files

### Deployment Steps

1. **Prepare the Model**:
   ```bash
   python scripts/convert.py --model-path ./models/my_model --formats pytorch onnx
   ```

2. **Start the API Server**:
   ```bash
   python scripts/deploy.py --model-path ./models/my_model --target local
   ```

3. **Access the API**:
   The API will be available at `http://localhost:8000`

### Configuration

Local deployment can be configured in the `deployment` section of the configuration file:

```yaml
deployment:
  default_target: "local"
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 1
```

## Cloud Deployment

### Prerequisites

- Cloud account (AWS, GCP, or Azure)
- Cloud CLI tools installed and configured
- Required cloud dependencies

### AWS Deployment

1. **Install AWS Dependencies**:
   ```bash
   pip install boto3
   ```

2. **Configure AWS CLI**:
   ```bash
   aws configure
   ```

3. **Deploy to AWS**:
   ```bash
   python scripts/deploy.py --model-path ./models/my_model --target cloud
   ```

### GCP Deployment

1. **Install GCP Dependencies**:
   ```bash
   pip install google-cloud-storage
   ```

2. **Authenticate with GCP**:
   ```bash
   gcloud auth login
   ```

3. **Deploy to GCP**:
   ```bash
   python scripts/deploy.py --model-path ./models/my_model --target cloud
   ```

### Azure Deployment

1. **Install Azure Dependencies**:
   ```bash
   pip install azure-storage-blob
   ```

2. **Authenticate with Azure**:
   ```bash
   az login
   ```

3. **Deploy to Azure**:
   ```bash
   python scripts/deploy.py --model-path ./models/my_model --target cloud
   ```

## Hybrid Deployment

Hybrid deployment combines local and cloud resources for optimal performance and cost.

### Configuration

```yaml
deployment:
  default_target: "hybrid"
  hybrid:
    local_workers: 2
    cloud_workers: 4
    load_balancing: "round-robin"
```

## Containerization

### Docker

The system includes Docker support for consistent deployment across environments.

1. **Build Docker Image**:
   ```bash
   docker build -t bangkong-llm .
   ```

2. **Run Docker Container**:
   ```bash
   docker run -p 8000:8000 bangkong-llm
   ```

### Docker Compose

For multi-container deployments:

```yaml
version: '3.8'
services:
  bangkong-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - BANGKONG_MODEL_PATH=/app/models/my_model
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster
- kubectl configured
- Helm (optional)

### Deployment Steps

1. **Create Kubernetes Manifests**:
   ```bash
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   ```

2. **Configure Ingress** (optional):
   ```bash
   kubectl apply -f kubernetes/ingress.yaml
   ```

### Kubernetes Manifests

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bangkong-llm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bangkong-llm
  template:
    metadata:
      labels:
        app: bangkong-llm
    spec:
      containers:
      - name: bangkong-llm
        image: bangkong-llm:latest
        ports:
        - containerPort: 8000
        env:
        - name: BANGKONG_MODEL_PATH
          value: "/models/my_model"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

#### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: bangkong-llm
spec:
  selector:
    app: bangkong-llm
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## API Documentation

Deployed models expose a REST API for inference:

### Endpoints

- `POST /generate`: Text generation
- `POST /complete`: Code completion
- `POST /classify`: Text classification
- `GET /health`: Health check

### Example Requests

#### Text Generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_length": 100}'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

## Monitoring and Scaling

### Health Monitoring

The deployment includes health monitoring endpoints:
- `/health`: Basic health check
- `/metrics`: Prometheus metrics
- `/status`: Detailed status information

### Auto-scaling

Cloud deployments support auto-scaling based on demand:

```yaml
autoscaling:
  min_replicas: 1
  max_replicas: 10
  target_cpu_utilization: 70
```

## Security Considerations

### Authentication

API endpoints can be protected with authentication:

```yaml
security:
  authentication: "token"
  authorization: "role-based"
```

### Encryption

- Data in transit: HTTPS/TLS
- Data at rest: Encryption at filesystem level
- Model weights: Optional encryption

### Network Security

- Firewall rules
- VPC isolation
- Private endpoints

## Performance Optimization

### Caching

Implement caching for frequently requested responses:

```yaml
caching:
  enabled: true
  ttl_seconds: 300
  max_size_mb: 100
```

### Load Balancing

Distribute requests across multiple instances:

```yaml
load_balancing:
  strategy: "round-robin"
  health_checks: true
```

### Resource Management

Optimize resource allocation:

```yaml
resources:
  cpu: "2"
  memory: "4Gi"
  gpu: "1"
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**:
   - Check model file integrity
   - Verify dependencies
   - Check file permissions

2. **Performance Issues**:
   - Monitor resource usage
   - Adjust batch sizes
   - Optimize model quantization

3. **Deployment Failures**:
   - Check logs for error messages
   - Verify configuration
   - Ensure dependencies are installed

### Logs and Monitoring

Check deployment logs for detailed error information:
```bash
kubectl logs deployment/bangkong-llm
```

## Best Practices

1. **Use Environment-Specific Configurations**: Different configurations for development, staging, and production
2. **Implement Health Checks**: Regular health monitoring
3. **Enable Logging**: Comprehensive logging for debugging
4. **Use Secure Communication**: HTTPS/TLS for API endpoints
5. **Implement Backup Strategies**: Regular backups of model files
6. **Monitor Resource Usage**: Track CPU, memory, and GPU usage
7. **Plan for Scaling**: Design for horizontal scaling from the start