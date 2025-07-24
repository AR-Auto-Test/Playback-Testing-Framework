package com.google.ar.sceneform.rendering;

import com.google.ar.core.annotations.UsedByNative;
import com.google.ar.sceneform.rendering.Texture;
import com.google.ar.sceneform.resources.SharedReference;
import com.google.ar.sceneform.utilities.AndroidPreconditions;

@UsedByNative("material_java_wrappers.h")
/* loaded from: classes.dex */
public class TextureInternalData extends SharedReference {
    private com.google.android.filament.Texture filamentTexture;
    private final Texture.Sampler sampler;

    @UsedByNative("material_java_wrappers.h")
    public TextureInternalData(com.google.android.filament.Texture texture, Texture.Sampler sampler) {
        this.filamentTexture = texture;
        this.sampler = sampler;
    }

    public com.google.android.filament.Texture getFilamentTexture() {
        com.google.android.filament.Texture texture = this.filamentTexture;
        if (texture != null) {
            return texture;
        }
        throw new IllegalStateException("Filament Texture is null.");
    }

    public Texture.Sampler getSampler() {
        return this.sampler;
    }

    @Override // com.google.ar.sceneform.resources.SharedReference
    public void onDispose() {
        AndroidPreconditions.checkUiThread();
        IEngine engine = EngineInstance.getEngine();
        com.google.android.filament.Texture texture = this.filamentTexture;
        this.filamentTexture = null;
        if (texture == null || engine == null || !engine.isValid()) {
            return;
        }
        engine.destroyTexture(texture);
    }
}