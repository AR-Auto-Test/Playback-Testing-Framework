package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.net.Uri;
import c.b.a.a.a;
import c.d.b.a.q.t;
import c.d.b.a.q.v;
import com.google.android.filament.Material;
import com.google.android.filament.MaterialInstance;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.LoadHelper;
import com.google.ar.sceneform.utilities.Preconditions;
import com.google.ar.sceneform.utilities.SceneformBufferUtils;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Function;
import java.util.function.Supplier;

/* loaded from: classes.dex */
public class Material {
    private static final String TAG = "Material";
    private final IMaterialInstance internalMaterialInstance;
    private final MaterialInternalData materialData;
    private final MaterialParameters materialParameters;

    /* loaded from: classes.dex */
    public static final class Builder {
        public com.google.android.filament.Material existingMaterial;
        private Callable<InputStream> inputStreamCreator;
        private Object registryId;
        public ByteBuffer sourceBuffer;

        private void checkPreconditions() {
            AndroidPreconditions.checkUiThread();
            if (!hasSource().booleanValue()) {
                throw new AssertionError("Material must have a source.");
            }
        }

        private com.google.android.filament.Material createFilamentMaterial(ByteBuffer byteBuffer) {
            try {
                return new Material.Builder().payload(byteBuffer, byteBuffer.limit()).build(EngineInstance.getEngine().getFilamentEngine());
            } catch (Exception e2) {
                throw new IllegalArgumentException("Unable to create material from source byte buffer.", e2);
            }
        }

        private Boolean hasSource() {
            return Boolean.valueOf((this.inputStreamCreator == null && this.sourceBuffer == null && this.existingMaterial == null) ? false : true);
        }

        public /* synthetic */ Material a(ByteBuffer byteBuffer) {
            return new Material(new MaterialInternalDataImpl(createFilamentMaterial(byteBuffer)));
        }

        /* JADX DEBUG: Type inference failed for r0v11. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<U>, java.util.concurrent.CompletableFuture<com.google.ar.sceneform.rendering.Material> */
        public CompletableFuture<Material> build() {
            CompletableFuture<Material> completableFuture;
            try {
                checkPreconditions();
                Object obj = this.registryId;
                if (obj != null && (completableFuture = ResourceManager.getInstance().getMaterialRegistry().get(obj)) != null) {
                    return completableFuture.thenApply((Function<? super Material, ? extends U>) t.f4353a);
                }
                ByteBuffer byteBuffer = this.sourceBuffer;
                if (byteBuffer != null) {
                    Material material = new Material(new MaterialInternalDataImpl(createFilamentMaterial(byteBuffer)));
                    if (obj != null) {
                        ResourceManager.getInstance().getMaterialRegistry().register(obj, CompletableFuture.completedFuture(material));
                    }
                    CompletableFuture<Material> completedFuture = CompletableFuture.completedFuture(material.makeCopy());
                    String str = Material.TAG;
                    FutureHelper.logOnException(str, completedFuture, "Unable to load Material registryId='" + obj + "'");
                    return completedFuture;
                }
                com.google.android.filament.Material material2 = this.existingMaterial;
                if (material2 != null) {
                    Material material3 = new Material(new MaterialInternalDataGltfImpl(material2));
                    if (obj != null) {
                        ResourceManager.getInstance().getMaterialRegistry().register(obj, CompletableFuture.completedFuture(material3.makeCopy()));
                    }
                    CompletableFuture<Material> completedFuture2 = CompletableFuture.completedFuture(material3);
                    String str2 = Material.TAG;
                    FutureHelper.logOnException(str2, completedFuture2, "Unable to load Material registryId='" + obj + "'");
                    return completedFuture2;
                }
                final Callable<InputStream> callable = this.inputStreamCreator;
                if (callable == null) {
                    CompletableFuture<Material> completableFuture2 = new CompletableFuture<>();
                    completableFuture2.completeExceptionally(new AssertionError("Input Stream Creator is null."));
                    return completableFuture2;
                }
                CompletableFuture<Material> thenApplyAsync = CompletableFuture.supplyAsync(new Supplier() { // from class: c.d.b.a.q.s
                    @Override // java.util.function.Supplier
                    public final Object get() {
                        try {
                            InputStream inputStream = (InputStream) callable.call();
                            ByteBuffer readStream = SceneformBufferUtils.readStream(inputStream);
                            if (inputStream != null) {
                                inputStream.close();
                            }
                            if (readStream != null) {
                                return readStream;
                            }
                            throw new IllegalStateException("Unable to read data from input stream.");
                        } catch (Exception e2) {
                            throw new CompletionException(e2);
                        }
                    }
                }, ThreadPools.getThreadPoolExecutor()).thenApplyAsync(new Function() { // from class: c.d.b.a.q.u
                    @Override // java.util.function.Function
                    public final Object apply(Object obj2) {
                        return Material.Builder.this.a((ByteBuffer) obj2);
                    }
                }, ThreadPools.getMainExecutor());
                if (obj != null) {
                    ResourceManager.getInstance().getMaterialRegistry().register(obj, thenApplyAsync);
                }
                return thenApplyAsync.thenApply((Function<? super Material, ? extends U>) v.f4355a);
            } catch (Throwable th) {
                CompletableFuture<Material> completableFuture3 = new CompletableFuture<>();
                completableFuture3.completeExceptionally(th);
                String str3 = Material.TAG;
                StringBuilder x = a.x("Unable to load Material registryId='");
                x.append(this.registryId);
                x.append("'");
                FutureHelper.logOnException(str3, completableFuture3, x.toString());
                return completableFuture3;
            }
        }

        public Builder setRegistryId(Object obj) {
            this.registryId = obj;
            return this;
        }

        public Builder setSource(ByteBuffer byteBuffer) {
            Preconditions.checkNotNull(byteBuffer, "Parameter \"materialBuffer\" was null.");
            this.inputStreamCreator = null;
            this.sourceBuffer = byteBuffer;
            return this;
        }

        private Builder() {
        }

        public Builder setSource(Context context, Uri uri) {
            Preconditions.checkNotNull(uri, "Parameter \"sourceUri\" was null.");
            this.registryId = uri;
            this.inputStreamCreator = LoadHelper.fromUri(context, uri);
            this.sourceBuffer = null;
            return this;
        }

        public Builder setSource(Context context, int i) {
            this.registryId = context.getResources().getResourceName(i);
            this.inputStreamCreator = LoadHelper.fromResource(context, i);
            this.sourceBuffer = null;
            return this;
        }

        public Builder setSource(Callable<InputStream> callable) {
            Preconditions.checkNotNull(callable, "Parameter \"sourceInputStreamCallable\" was null.");
            this.inputStreamCreator = callable;
            this.sourceBuffer = null;
            return this;
        }
    }

    /* loaded from: classes.dex */
    public static final class CleanupCallback implements Runnable {
        private final IMaterialInstance materialInstance;
        private final MaterialInternalData materialInternalData;

        public CleanupCallback(IMaterialInstance iMaterialInstance, MaterialInternalData materialInternalData) {
            this.materialInstance = iMaterialInstance;
            this.materialInternalData = materialInternalData;
        }

        @Override // java.lang.Runnable
        public void run() {
            AndroidPreconditions.checkUiThread();
            IMaterialInstance iMaterialInstance = this.materialInstance;
            if (iMaterialInstance != null) {
                iMaterialInstance.dispose();
            }
            MaterialInternalData materialInternalData = this.materialInternalData;
            if (materialInternalData != null) {
                materialInternalData.release();
            }
        }
    }

    /* loaded from: classes.dex */
    public interface IMaterialInstance {
        void dispose();

        MaterialInstance getInstance();

        boolean isValidInstance();
    }

    /* loaded from: classes.dex */
    public static class InternalGltfMaterialInstance implements IMaterialInstance {
        public MaterialInstance instance;

        @Override // com.google.ar.sceneform.rendering.Material.IMaterialInstance
        public void dispose() {
        }

        @Override // com.google.ar.sceneform.rendering.Material.IMaterialInstance
        public MaterialInstance getInstance() {
            return (MaterialInstance) Preconditions.checkNotNull(this.instance);
        }

        @Override // com.google.ar.sceneform.rendering.Material.IMaterialInstance
        public boolean isValidInstance() {
            return this.instance != null;
        }

        public void setMaterialInstance(MaterialInstance materialInstance) {
            this.instance = materialInstance;
        }
    }

    /* loaded from: classes.dex */
    public static class InternalMaterialInstance implements IMaterialInstance {
        public final MaterialInstance instance;

        public InternalMaterialInstance(MaterialInstance materialInstance) {
            this.instance = materialInstance;
        }

        @Override // com.google.ar.sceneform.rendering.Material.IMaterialInstance
        public void dispose() {
            IEngine engine = EngineInstance.getEngine();
            if (engine == null || !engine.isValid()) {
                return;
            }
            engine.destroyMaterialInstance(this.instance);
        }

        @Override // com.google.ar.sceneform.rendering.Material.IMaterialInstance
        public MaterialInstance getInstance() {
            return this.instance;
        }

        @Override // com.google.ar.sceneform.rendering.Material.IMaterialInstance
        public boolean isValidInstance() {
            return this.instance != null;
        }
    }

    public Material(MaterialInternalData materialInternalData) {
        this.materialParameters = new MaterialParameters();
        this.materialData = materialInternalData;
        materialInternalData.retain();
        if (materialInternalData instanceof MaterialInternalDataImpl) {
            this.internalMaterialInstance = new InternalMaterialInstance(materialInternalData.getFilamentMaterial().createInstance());
        } else {
            this.internalMaterialInstance = new InternalGltfMaterialInstance();
        }
        ResourceManager.getInstance().getMaterialCleanupRegistry().register(this, new CleanupCallback(this.internalMaterialInstance, materialInternalData));
    }

    public static Builder builder() {
        AndroidPreconditions.checkMinAndroidApiLevel();
        return new Builder();
    }

    private static native Object nGetMaterialParameters(long j, int i);

    public void copyMaterialParameters(MaterialParameters materialParameters) {
        this.materialParameters.copyFrom(materialParameters);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public ExternalTexture getExternalTexture(String str) {
        return this.materialParameters.getExternalTexture(str);
    }

    public MaterialInstance getFilamentMaterialInstance() {
        if (this.internalMaterialInstance.isValidInstance()) {
            return this.internalMaterialInstance.getInstance();
        }
        throw new AssertionError("Filament Material Instance is null.");
    }

    public Material makeCopy() {
        return new Material(this);
    }

    public void setBoolean(String str, boolean z) {
        this.materialParameters.setBoolean(str, z);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setBoolean2(String str, boolean z, boolean z2) {
        this.materialParameters.setBoolean2(str, z, z2);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setBoolean3(String str, boolean z, boolean z2, boolean z3) {
        this.materialParameters.setBoolean3(str, z, z2, z3);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setBoolean4(String str, boolean z, boolean z2, boolean z3, boolean z4) {
        this.materialParameters.setBoolean4(str, z, z2, z3, z4);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setExternalTexture(String str, ExternalTexture externalTexture) {
        this.materialParameters.setExternalTexture(str, externalTexture);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setFloat(String str, float f2) {
        this.materialParameters.setFloat(str, f2);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setFloat2(String str, float f2, float f3) {
        this.materialParameters.setFloat2(str, f2, f3);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setFloat3(String str, float f2, float f3, float f4) {
        this.materialParameters.setFloat3(str, f2, f3, f4);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setFloat4(String str, float f2, float f3, float f4, float f5) {
        this.materialParameters.setFloat4(str, f2, f3, f4, f5);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setInt(String str, int i) {
        this.materialParameters.setInt(str, i);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setInt2(String str, int i, int i2) {
        this.materialParameters.setInt2(str, i, i2);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setInt3(String str, int i, int i2, int i3) {
        this.materialParameters.setInt3(str, i, i2, i3);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setInt4(String str, int i, int i2, int i3, int i4) {
        this.materialParameters.setInt4(str, i, i2, i3, i4);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setTexture(String str, Texture texture) {
        this.materialParameters.setTexture(str, texture);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void updateGltfMaterialInstance(MaterialInstance materialInstance) {
        IMaterialInstance iMaterialInstance = this.internalMaterialInstance;
        if (iMaterialInstance instanceof InternalGltfMaterialInstance) {
            ((InternalGltfMaterialInstance) iMaterialInstance).setMaterialInstance(materialInstance);
            this.materialParameters.applyTo(materialInstance);
        }
    }

    public void setFloat3(String str, Vector3 vector3) {
        this.materialParameters.setFloat3(str, vector3);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setFloat4(String str, Color color) {
        this.materialParameters.setFloat4(str, color.r, color.f5628g, color.f5627b, color.f5626a);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    public void setFloat3(String str, Color color) {
        this.materialParameters.setFloat3(str, color.r, color.f5628g, color.f5627b);
        if (this.internalMaterialInstance.isValidInstance()) {
            this.materialParameters.applyTo(this.internalMaterialInstance.getInstance());
        }
    }

    private Material(Material material) {
        this(material.materialData);
        copyMaterialParameters(material.materialParameters);
    }
}