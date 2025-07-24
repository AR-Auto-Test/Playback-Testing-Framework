package com.google.ar.sceneform.rendering;

import com.google.ar.sceneform.resources.ResourceHolder;
import com.google.ar.sceneform.resources.ResourceRegistry;
import java.util.ArrayList;
import java.util.Iterator;

/* loaded from: classes.dex */
public class ResourceManager {
    private static ResourceManager instance;
    private final CleanupRegistry<CameraStream> cameraStreamCleanupRegistry;
    private final CleanupRegistry<ExternalTexture> externalTextureCleanupRegistry;
    private final CleanupRegistry<Material> materialCleanupRegistry;
    private final ResourceRegistry<Material> materialRegistry;
    private final ResourceRegistry<ModelRenderable> modelRenderableRegistry;
    private final CleanupRegistry<RenderableInstance> renderableInstanceCleanupRegistry;
    private final ArrayList<ResourceHolder> resourceHolders = new ArrayList<>();
    private final CleanupRegistry<SimpleCameraStream> simpleCameraStreamCleanupRegistry;
    private final CleanupRegistry<Texture> textureCleanupRegistry;
    private final ResourceRegistry<Texture> textureRegistry;
    private final ResourceRegistry<ViewRenderable> viewRenderableRegistry;

    private ResourceManager() {
        ResourceRegistry<Texture> resourceRegistry = new ResourceRegistry<>();
        this.textureRegistry = resourceRegistry;
        ResourceRegistry<Material> resourceRegistry2 = new ResourceRegistry<>();
        this.materialRegistry = resourceRegistry2;
        ResourceRegistry<ModelRenderable> resourceRegistry3 = new ResourceRegistry<>();
        this.modelRenderableRegistry = resourceRegistry3;
        this.viewRenderableRegistry = new ResourceRegistry<>();
        CleanupRegistry<CameraStream> cleanupRegistry = new CleanupRegistry<>();
        this.cameraStreamCleanupRegistry = cleanupRegistry;
        CleanupRegistry<SimpleCameraStream> cleanupRegistry2 = new CleanupRegistry<>();
        this.simpleCameraStreamCleanupRegistry = cleanupRegistry2;
        CleanupRegistry<ExternalTexture> cleanupRegistry3 = new CleanupRegistry<>();
        this.externalTextureCleanupRegistry = cleanupRegistry3;
        CleanupRegistry<Material> cleanupRegistry4 = new CleanupRegistry<>();
        this.materialCleanupRegistry = cleanupRegistry4;
        CleanupRegistry<RenderableInstance> cleanupRegistry5 = new CleanupRegistry<>();
        this.renderableInstanceCleanupRegistry = cleanupRegistry5;
        CleanupRegistry<Texture> cleanupRegistry6 = new CleanupRegistry<>();
        this.textureCleanupRegistry = cleanupRegistry6;
        addResourceHolder(resourceRegistry);
        addResourceHolder(resourceRegistry2);
        addResourceHolder(resourceRegistry3);
        addViewRenderableRegistry();
        addResourceHolder(cleanupRegistry);
        addResourceHolder(cleanupRegistry2);
        addResourceHolder(cleanupRegistry3);
        addResourceHolder(cleanupRegistry4);
        addResourceHolder(cleanupRegistry5);
        addResourceHolder(cleanupRegistry6);
    }

    private void addViewRenderableRegistry() {
        addResourceHolder(this.viewRenderableRegistry);
    }

    public static ResourceManager getInstance() {
        if (instance == null) {
            instance = new ResourceManager();
        }
        return instance;
    }

    public void addResourceHolder(ResourceHolder resourceHolder) {
        this.resourceHolders.add(resourceHolder);
    }

    public void destroyAllResources() {
        Iterator<ResourceHolder> it = this.resourceHolders.iterator();
        while (it.hasNext()) {
            it.next().destroyAllResources();
        }
    }

    public CleanupRegistry<CameraStream> getCameraStreamCleanupRegistry() {
        return this.cameraStreamCleanupRegistry;
    }

    public CleanupRegistry<ExternalTexture> getExternalTextureCleanupRegistry() {
        return this.externalTextureCleanupRegistry;
    }

    public CleanupRegistry<Material> getMaterialCleanupRegistry() {
        return this.materialCleanupRegistry;
    }

    public ResourceRegistry<Material> getMaterialRegistry() {
        return this.materialRegistry;
    }

    public ResourceRegistry<ModelRenderable> getModelRenderableRegistry() {
        return this.modelRenderableRegistry;
    }

    public CleanupRegistry<RenderableInstance> getRenderableInstanceCleanupRegistry() {
        return this.renderableInstanceCleanupRegistry;
    }

    public CleanupRegistry<SimpleCameraStream> getSimpleCameraStreamCleanupRegistry() {
        return this.simpleCameraStreamCleanupRegistry;
    }

    public CleanupRegistry<Texture> getTextureCleanupRegistry() {
        return this.textureCleanupRegistry;
    }

    public ResourceRegistry<Texture> getTextureRegistry() {
        return this.textureRegistry;
    }

    public ResourceRegistry<ViewRenderable> getViewRenderableRegistry() {
        return this.viewRenderableRegistry;
    }

    public long reclaimReleasedResources() {
        Iterator<ResourceHolder> it = this.resourceHolders.iterator();
        long j = 0;
        while (it.hasNext()) {
            j += it.next().reclaimReleasedResources();
        }
        return j;
    }
}