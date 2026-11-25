FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /source
COPY src/ src/
COPY modules/ObjectDetectionYOLOv5Net/ modules/ObjectDetectionYOLOv5Net/
RUN dotnet publish src/server/Server.csproj -c Release -o /out/server
RUN dotnet publish modules/ObjectDetectionYOLOv5Net/ObjectDetectionYOLOv5Net.csproj -c Release -p:GpuType=CPU -o /out/module/bin

FROM mcr.microsoft.com/dotnet/aspnet:10.0
RUN apt-get update && apt-get install -y --no-install-recommends libfontconfig1 libfreetype6 curl procps && rm -rf /var/lib/apt/lists/*
WORKDIR /app/server
COPY --from=build /out/server/ /app/server/
COPY --from=build /out/module/bin/ /app/preinstalled-modules/ObjectDetectionYOLOv5Net/bin/
COPY modules/ObjectDetectionYOLOv5Net/modulesettings.json /app/preinstalled-modules/ObjectDetectionYOLOv5Net/modulesettings.json
COPY modules/ObjectDetectionYOLOv5Net/appsettings.json /app/preinstalled-modules/ObjectDetectionYOLOv5Net/appsettings.json
COPY modules/ObjectDetectionYOLOv5Net/explore.html /app/preinstalled-modules/ObjectDetectionYOLOv5Net/explore.html
COPY src/scripts/ /app/scripts/
COPY devops/docker-experimental/module-settings.json /app/preinstalled-modules/ObjectDetectionYOLOv5Net/modulesettings.docker.json
RUN mkdir -p /app/modules /app/preinstalled-modules/ObjectDetectionYOLOv5Net/assets /app/preinstalled-modules/ObjectDetectionYOLOv5Net/custom-models /etc/codeproject/ai
ENV CPAI_PORT=32168 \
    ServerOptions__AllowInternetAccess=false \
    ServerOptions__DisableLegacyPort=true \
    ModuleOptions__InstallInitialModules=false \
    MeshOptions__Enable=false
EXPOSE 32168
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -fsS http://127.0.0.1:32168/ || exit 1
ENTRYPOINT ["dotnet", "CodeProject.AI.Server.dll"]
