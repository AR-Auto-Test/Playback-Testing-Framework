package com.google.ar.sceneform.rendering;

import android.content.Context;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.MaterialFactory;
import com.google.ar.sceneform.rendering.RenderingResources;
import com.google.ar.sceneform.rendering.Texture;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

/* loaded from: classes.dex */
public final class MaterialFactory {
    private static final float DEFAULT_METALLIC_PROPERTY = 0.0f;
    private static final float DEFAULT_REFLECTANCE_PROPERTY = 0.5f;
    private static final float DEFAULT_ROUGHNESS_PROPERTY = 0.4f;
    public static final String MATERIAL_COLOR = "color";
    public static final String MATERIAL_METALLIC = "metallic";
    public static final String MATERIAL_REFLECTANCE = "reflectance";
    public static final String MATERIAL_ROUGHNESS = "roughness";
    public static final String MATERIAL_TEXTURE = "texture";

    private static void applyDefaultPbrParams(Material material) {
        material.setFloat(MATERIAL_METALLIC, 0.0f);
        material.setFloat(MATERIAL_ROUGHNESS, DEFAULT_ROUGHNESS_PROPERTY);
        material.setFloat(MATERIAL_REFLECTANCE, 0.5f);
    }

    public static /* synthetic */ Material lambda$makeOpaqueWithColor$0(Color color, Material material) {
        material.setFloat3("color", color);
        applyDefaultPbrParams(material);
        return material;
    }

    public static /* synthetic */ Material lambda$makeOpaqueWithTexture$2(Texture texture, Material material) {
        material.setTexture("texture", texture);
        applyDefaultPbrParams(material);
        return material;
    }

    public static /* synthetic */ Material lambda$makeTransparentWithColor$1(Color color, Material material) {
        material.setFloat4("color", color);
        applyDefaultPbrParams(material);
        return material;
    }

    public static /* synthetic */ Material lambda$makeTransparentWithTexture$3(Texture texture, Material material) {
        material.setTexture("texture", texture);
        applyDefaultPbrParams(material);
        return material;
    }

    /* JADX DEBUG: Type inference failed for r2v3. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<U>, java.util.concurrent.CompletableFuture<com.google.ar.sceneform.rendering.Material> */
    public static CompletableFuture<Material> makeOpaqueWithColor(Context context, final Color color) {
        return Material.builder().setSource(context, RenderingResources.GetSceneformResource(context, RenderingResources.Resource.OPAQUE_COLORED_MATERIAL)).build().thenApply(new Function() { // from class: c.d.b.a.q.x
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Material material = (Material) obj;
                MaterialFactory.lambda$makeOpaqueWithColor$0(Color.this, material);
                return material;
            }
        });
    }

    /* JADX DEBUG: Type inference failed for r2v3. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<U>, java.util.concurrent.CompletableFuture<com.google.ar.sceneform.rendering.Material> */
    public static CompletableFuture<Material> makeOpaqueWithTexture(Context context, final Texture texture) {
        return Material.builder().setSource(context, RenderingResources.GetSceneformResource(context, RenderingResources.Resource.OPAQUE_TEXTURED_MATERIAL)).build().thenApply(new Function() { // from class: c.d.b.a.q.w
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Material material = (Material) obj;
                MaterialFactory.lambda$makeOpaqueWithTexture$2(Texture.this, material);
                return material;
            }
        });
    }

    /* JADX DEBUG: Type inference failed for r2v3. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<U>, java.util.concurrent.CompletableFuture<com.google.ar.sceneform.rendering.Material> */
    public static CompletableFuture<Material> makeTransparentWithColor(Context context, final Color color) {
        return Material.builder().setSource(context, RenderingResources.GetSceneformResource(context, RenderingResources.Resource.TRANSPARENT_COLORED_MATERIAL)).build().thenApply(new Function() { // from class: c.d.b.a.q.y
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Material material = (Material) obj;
                MaterialFactory.lambda$makeTransparentWithColor$1(Color.this, material);
                return material;
            }
        });
    }

    /* JADX DEBUG: Type inference failed for r2v3. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<U>, java.util.concurrent.CompletableFuture<com.google.ar.sceneform.rendering.Material> */
    public static CompletableFuture<Material> makeTransparentWithTexture(Context context, final Texture texture) {
        return Material.builder().setSource(context, RenderingResources.GetSceneformResource(context, RenderingResources.Resource.TRANSPARENT_TEXTURED_MATERIAL)).build().thenApply(new Function() { // from class: c.d.b.a.q.z
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Material material = (Material) obj;
                MaterialFactory.lambda$makeTransparentWithTexture$3(Texture.this, material);
                return material;
            }
        });
    }
}