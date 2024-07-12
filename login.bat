@echo off
setlocal
for /f "tokens=*" %%i in ('aws ecr-public get-login-password --region us-east-1') do (echo %%i | docker login --username AWS --password-stdin public.ecr.aws/v1i9d3i6)
endlocal
