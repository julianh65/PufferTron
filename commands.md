to go to this current directory do 
cd /workspaces/PufferTron

to hook into it from terminal
hostname (run this in termianl in vscode)
docker exec -it boring_satoshi ls -la /workspaces/PufferTron



mental model of image vs container
container is running recipe with a filesystem and process
dev container - vs code starts a container for my folder, mounts code inside it and runs vs code
