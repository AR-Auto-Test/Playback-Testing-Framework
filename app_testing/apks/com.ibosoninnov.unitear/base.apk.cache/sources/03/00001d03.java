package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.net.Uri;
import c.b.a.a.a;
import com.google.ar.sceneform.collision.Box;
import com.google.ar.sceneform.collision.CollisionShape;
import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.resources.ResourceRegistry;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.ChangeId;
import com.google.ar.sceneform.utilities.LoadHelper;
import com.google.ar.sceneform.utilities.Preconditions;
import com.google.common.net.HttpHeaders;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/* loaded from: classes.dex */
public abstract class Renderable {
    public static final int DEFAULT_ANIMATION_FRAME_RATE = 24;
    private static final long DEFAULT_MAX_STALE_CACHE = TimeUnit.DAYS.toSeconds(14);
    public static final int RENDER_PRIORITY_DEFAULT = 4;
    public static final int RENDER_PRIORITY_FIRST = 0;
    public static final int RENDER_PRIORITY_LAST = 7;
    private int animationFrameRate;
    public boolean asyncLoadEnabled;
    private final ChangeId changeId;
    public CollisionShape collisionShape;
    private boolean isShadowCaster;
    private boolean isShadowReceiver;
    private final ArrayList<Material> materialBindings;
    private final ArrayList<String> materialNames;
    private int renderPriority;
    private final IRenderableInternalData renderableData;

    public Renderable(Builder<? extends Renderable, ? extends Builder<?, ?>> builder) {
        this.materialBindings = new ArrayList<>();
        this.materialNames = new ArrayList<>();
        this.renderPriority = 4;
        this.isShadowCaster = true;
        this.isShadowReceiver = true;
        this.changeId = new ChangeId();
        Preconditions.checkNotNull(builder, "Parameter \"builder\" was null.");
        if (!((Builder) builder).isFilamentAsset) {
            if (((Builder) builder).isGltf) {
                this.renderableData = createRenderableInternalGltfData();
            } else {
                this.renderableData = new RenderableInternalData();
            }
        } else {
            this.renderableData = new RenderableInternalFilamentAssetData();
        }
        if (((Builder) builder).definition != null) {
            updateFromDefinition(((Builder) builder).definition);
        }
        this.asyncLoadEnabled = ((Builder) builder).asyncLoadEnabled;
        this.animationFrameRate = ((Builder) builder).animationFrameRate;
    }

    private IRenderableInternalData createRenderableInternalGltfData() {
        return null;
    }

    private IllegalArgumentException makeSubmeshOutOfRangeException(int i) {
        StringBuilder y = a.y("submeshIndex (", i, ") is out of range. It must be less than the submeshCount (");
        y.append(getSubmeshCount());
        y.append(").");
        return new IllegalArgumentException(y.toString());
    }

    public void attachToRenderer(Renderer renderer) {
    }

    public RenderableInstance createInstance(TransformProvider transformProvider) {
        return new RenderableInstance(transformProvider, this);
    }

    public void detatchFromRenderer() {
    }

    public int getAnimationFrameRate() {
        return this.animationFrameRate;
    }

    public CollisionShape getCollisionShape() {
        return this.collisionShape;
    }

    public Matrix getFinalModelMatrix(Matrix matrix) {
        Preconditions.checkNotNull(matrix, "Parameter \"originalMatrix\" was null.");
        return matrix;
    }

    public ChangeId getId() {
        return this.changeId;
    }

    public Material getMaterial() {
        return getMaterial(0);
    }

    public ArrayList<Material> getMaterialBindings() {
        return this.materialBindings;
    }

    public ArrayList<String> getMaterialNames() {
        return this.materialNames;
    }

    public int getRenderPriority() {
        return this.renderPriority;
    }

    public IRenderableInternalData getRenderableData() {
        return this.renderableData;
    }

    public int getSubmeshCount() {
        return this.renderableData.getMeshes().size();
    }

    public String getSubmeshName(int i) {
        Preconditions.checkState(this.materialNames.size() == this.materialBindings.size());
        if (i >= 0 && i < this.materialNames.size()) {
            return this.materialNames.get(i);
        }
        throw makeSubmeshOutOfRangeException(i);
    }

    public boolean isShadowCaster() {
        return this.isShadowCaster;
    }

    public boolean isShadowReceiver() {
        return this.isShadowReceiver;
    }

    public abstract Renderable makeCopy();

    public void prepareForDraw() {
        if (getRenderableData() instanceof RenderableInternalFilamentAssetData) {
            ((RenderableInternalFilamentAssetData) getRenderableData()).resourceLoader.asyncUpdateLoad();
        }
    }

    public void setCollisionShape(CollisionShape collisionShape) {
        this.collisionShape = collisionShape;
        this.changeId.update();
    }

    public void setMaterial(Material material) {
        setMaterial(0, material);
    }

    public void setRenderPriority(int i) {
        this.renderPriority = Math.min(7, Math.max(0, i));
        this.changeId.update();
    }

    public void setShadowCaster(boolean z) {
        this.isShadowCaster = z;
        this.changeId.update();
    }

    public void setShadowReceiver(boolean z) {
        this.isShadowReceiver = z;
        this.changeId.update();
    }

    public void updateFromDefinition(RenderableDefinition renderableDefinition) {
        Preconditions.checkState(!renderableDefinition.getSubmeshes().isEmpty());
        this.changeId.update();
        renderableDefinition.applyDefinitionToData(this.renderableData, this.materialBindings, this.materialNames);
        this.collisionShape = new Box(this.renderableData.getSizeAabb(), this.renderableData.getCenterAabb());
    }

    public Material getMaterial(int i) {
        if (i < this.materialBindings.size()) {
            return this.materialBindings.get(i);
        }
        throw makeSubmeshOutOfRangeException(i);
    }

    public void setMaterial(int i, Material material) {
        if (i < this.materialBindings.size()) {
            this.materialBindings.set(i, material);
            this.changeId.update();
            return;
        }
        throw makeSubmeshOutOfRangeException(i);
    }

    /* loaded from: classes.dex */
    public static abstract class Builder<T extends Renderable, B extends Builder<T, B>> {
        private LoadGltfListener loadGltfListener;
        public Object registryId = null;
        public Context context = null;
        private Uri sourceUri = null;
        private Callable<InputStream> inputStreamCreator = null;
        private RenderableDefinition definition = null;
        private boolean isGltf = false;
        private boolean isFilamentAsset = false;
        private boolean asyncLoadEnabled = false;
        private Function<String, Uri> uriResolver = null;
        private byte[] materialsBytes = null;
        private int animationFrameRate = 24;

        private CompletableFuture<T> loadRenderableFromFilamentGltf(Context context, T t) {
            return new LoadRenderableFromFilamentGltfTask(t, context, (Uri) Preconditions.checkNotNull(this.sourceUri), this.uriResolver).downloadAndProcessRenderable((Callable) Preconditions.checkNotNull(this.inputStreamCreator));
        }

        private CompletableFuture<T> loadRenderableFromGltf(Context context, T t, byte[] bArr) {
            return null;
        }

        private void setCachingEnabled(Context context) {
        }

        private B setRemoteSourceHelper(Context context, Uri uri, boolean z) {
            Preconditions.checkNotNull(uri);
            this.sourceUri = uri;
            this.context = context;
            this.registryId = uri;
            if (z) {
                setCachingEnabled(context);
            }
            HashMap hashMap = new HashMap();
            if (!z) {
                hashMap.put(HttpHeaders.CACHE_CONTROL, "no-cache");
            } else {
                StringBuilder x = a.x("max-stale=");
                x.append(Renderable.DEFAULT_MAX_STALE_CACHE);
                hashMap.put(HttpHeaders.CACHE_CONTROL, x.toString());
            }
            this.inputStreamCreator = LoadHelper.fromUri(context, (Uri) Preconditions.checkNotNull(this.sourceUri), hashMap);
            return getSelf();
        }

        public CompletableFuture<T> build() {
            CompletableFuture<T> downloadAndProcessRenderable;
            CompletableFuture<T> completableFuture;
            try {
                checkPreconditions();
                Object obj = this.registryId;
                if (obj != null && (completableFuture = getRenderableRegistry().get(obj)) != null) {
                    return (CompletableFuture<T>) completableFuture.thenApply(new Function() { // from class: c.d.b.a.q.e0
                        @Override // java.util.function.Function
                        public final Object apply(Object obj2) {
                            return (Renderable) Renderable.Builder.this.getRenderableClass().cast(((Renderable) obj2).makeCopy());
                        }
                    });
                }
                T makeRenderable = makeRenderable();
                if (this.definition != null) {
                    return CompletableFuture.completedFuture(makeRenderable);
                }
                Callable<InputStream> callable = this.inputStreamCreator;
                if (callable == null) {
                    CompletableFuture<T> completableFuture2 = new CompletableFuture<>();
                    completableFuture2.completeExceptionally(new AssertionError("Input Stream Creator is null."));
                    String simpleName = getRenderableClass().getSimpleName();
                    FutureHelper.logOnException(simpleName, completableFuture2, "Unable to load Renderable registryId='" + obj + "'");
                    return completableFuture2;
                }
                if (this.isFilamentAsset) {
                    Context context = this.context;
                    if (context != null) {
                        downloadAndProcessRenderable = loadRenderableFromFilamentGltf(context, makeRenderable);
                    } else {
                        throw new AssertionError("Gltf Renderable.Builder must have a valid context.");
                    }
                } else if (this.isGltf) {
                    Context context2 = this.context;
                    if (context2 != null) {
                        downloadAndProcessRenderable = loadRenderableFromGltf(context2, makeRenderable, this.materialsBytes);
                    } else {
                        throw new AssertionError("Gltf Renderable.Builder must have a valid context.");
                    }
                } else {
                    downloadAndProcessRenderable = new LoadRenderableFromSfbTask(makeRenderable, this.sourceUri).downloadAndProcessRenderable(callable);
                }
                if (obj != null) {
                    getRenderableRegistry().register(obj, downloadAndProcessRenderable);
                }
                String simpleName2 = getRenderableClass().getSimpleName();
                FutureHelper.logOnException(simpleName2, downloadAndProcessRenderable, "Unable to load Renderable registryId='" + obj + "'");
                return (CompletableFuture<T>) downloadAndProcessRenderable.thenApply(new Function() { // from class: c.d.b.a.q.d0
                    @Override // java.util.function.Function
                    public final Object apply(Object obj2) {
                        return (Renderable) Renderable.Builder.this.getRenderableClass().cast(((Renderable) obj2).makeCopy());
                    }
                });
            } catch (Throwable th) {
                CompletableFuture<T> completableFuture3 = new CompletableFuture<>();
                completableFuture3.completeExceptionally(th);
                String simpleName3 = getRenderableClass().getSimpleName();
                StringBuilder x = a.x("Unable to load Renderable registryId='");
                x.append(this.registryId);
                x.append("'");
                FutureHelper.logOnException(simpleName3, completableFuture3, x.toString());
                return completableFuture3;
            }
        }

        public void checkPreconditions() {
            AndroidPreconditions.checkUiThread();
            if (!hasSource().booleanValue()) {
                throw new AssertionError("ModelRenderable must have a source.");
            }
        }

        public abstract Class<T> getRenderableClass();

        public abstract ResourceRegistry<T> getRenderableRegistry();

        public abstract B getSelf();

        public Boolean hasSource() {
            return Boolean.valueOf((this.sourceUri == null && this.inputStreamCreator == null && this.definition == null) ? false : true);
        }

        public abstract T makeRenderable();

        public B setAnimationFrameRate(int i) {
            this.animationFrameRate = i;
            return getSelf();
        }

        public B setAsyncLoadEnabled(boolean z) {
            this.asyncLoadEnabled = z;
            return getSelf();
        }

        public B setIsFilamentGltf(boolean z) {
            this.isFilamentAsset = z;
            return getSelf();
        }

        public B setRegistryId(Object obj) {
            this.registryId = obj;
            return getSelf();
        }

        public B setSource(Context context, Uri uri, boolean z) {
            return null;
        }

        public B setSource(Context context, Callable<InputStream> callable) {
            Preconditions.checkNotNull(callable);
            this.sourceUri = null;
            this.inputStreamCreator = callable;
            this.context = context;
            return getSelf();
        }

        public B setSource(Context context, Uri uri) {
            return setRemoteSourceHelper(context, uri, true);
        }

        public B setSource(Context context, int i) {
            this.inputStreamCreator = LoadHelper.fromResource(context, i);
            this.context = context;
            Uri resourceToUri = LoadHelper.resourceToUri(context, i);
            this.sourceUri = resourceToUri;
            this.registryId = resourceToUri;
            return getSelf();
        }

        public B setSource(RenderableDefinition renderableDefinition) {
            this.definition = renderableDefinition;
            this.registryId = null;
            this.sourceUri = null;
            return getSelf();
        }
    }

    public Renderable(Renderable renderable) {
        this.materialBindings = new ArrayList<>();
        this.materialNames = new ArrayList<>();
        this.renderPriority = 4;
        this.isShadowCaster = true;
        this.isShadowReceiver = true;
        this.changeId = new ChangeId();
        if (!renderable.getId().isEmpty()) {
            this.renderableData = renderable.renderableData;
            Preconditions.checkState(renderable.materialNames.size() == renderable.materialBindings.size());
            for (int i = 0; i < renderable.materialBindings.size(); i++) {
                this.materialBindings.add(renderable.materialBindings.get(i).makeCopy());
                this.materialNames.add(renderable.materialNames.get(i));
            }
            this.renderPriority = renderable.renderPriority;
            this.isShadowCaster = renderable.isShadowCaster;
            this.isShadowReceiver = renderable.isShadowReceiver;
            CollisionShape collisionShape = renderable.collisionShape;
            if (collisionShape != null) {
                this.collisionShape = collisionShape.makeCopy();
            }
            this.asyncLoadEnabled = renderable.asyncLoadEnabled;
            this.animationFrameRate = renderable.animationFrameRate;
            this.changeId.update();
            return;
        }
        throw new AssertionError("Cannot copy uninitialized Renderable.");
    }
}