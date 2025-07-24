package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.net.Uri;
import c.b.a.a.a;
import com.google.android.filament.gltfio.ResourceLoader;
import com.google.ar.sceneform.rendering.LoadRenderableFromFilamentGltfTask;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.utilities.Preconditions;
import com.google.ar.sceneform.utilities.SceneformBufferUtils;
import java.io.InputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Function;
import java.util.function.Supplier;

/* loaded from: classes.dex */
public class LoadRenderableFromFilamentGltfTask<T extends Renderable> {
    private static final String TAG = "LoadRenderableFromFilamentGltfTask";
    private final T renderable;
    private final RenderableInternalFilamentAssetData renderableData;

    public LoadRenderableFromFilamentGltfTask(T t, Context context, final Uri uri, final Function<String, Uri> function) {
        this.renderable = t;
        IRenderableInternalData renderableData = t.getRenderableData();
        if (renderableData instanceof RenderableInternalFilamentAssetData) {
            RenderableInternalFilamentAssetData renderableInternalFilamentAssetData = (RenderableInternalFilamentAssetData) renderableData;
            this.renderableData = renderableInternalFilamentAssetData;
            renderableInternalFilamentAssetData.resourceLoader = new ResourceLoader(EngineInstance.getEngine().getFilamentEngine());
            renderableInternalFilamentAssetData.urlResolver = new Function() { // from class: c.d.b.a.q.i
                @Override // java.util.function.Function
                public final Object apply(Object obj) {
                    return LoadRenderableFromFilamentGltfTask.getUriFromMissingResource(uri, (String) obj, function);
                }
            };
            renderableInternalFilamentAssetData.context = context.getApplicationContext();
            t.getId().update();
            return;
        }
        StringBuilder x = a.x("Expected task type ");
        x.append(TAG);
        throw new IllegalStateException(x.toString());
    }

    public static Uri getUriFromMissingResource(Uri uri, String str, Function<String, Uri> function) {
        if (function != null) {
            return function.apply(str);
        }
        if (str.startsWith("/")) {
            str = str.substring(1);
        }
        Uri parse = Uri.parse(Uri.decode(str));
        if (parse.getScheme() == null) {
            return Uri.parse(Uri.decode(URI.create(Uri.parse(Uri.decode(uri.toString())).buildUpon().appendPath("..").appendPath((String) Preconditions.checkNotNull(parse.getPath())).build().toString()).normalize().toString()));
        }
        throw new AssertionError(String.format("Resource path contains a scheme but should be relative, uri: (%s)", parse));
    }

    public /* synthetic */ Renderable a(byte[] bArr) {
        RenderableInternalFilamentAssetData renderableInternalFilamentAssetData = this.renderableData;
        boolean z = false;
        if (bArr[0] == 103 && bArr[1] == 108 && bArr[2] == 84 && bArr[3] == 70) {
            z = true;
        }
        renderableInternalFilamentAssetData.isGltfBinary = z;
        renderableInternalFilamentAssetData.gltfByteBuffer = ByteBuffer.wrap(bArr);
        return this.renderable;
    }

    public CompletableFuture<T> downloadAndProcessRenderable(final Callable<InputStream> callable) {
        return CompletableFuture.supplyAsync(new Supplier() { // from class: c.d.b.a.q.h
            @Override // java.util.function.Supplier
            public final Object get() {
                try {
                    return SceneformBufferUtils.inputStreamCallableToByteArray(callable);
                } catch (Exception e2) {
                    throw new CompletionException(e2);
                }
            }
        }, ThreadPools.getThreadPoolExecutor()).thenApplyAsync(new Function() { // from class: c.d.b.a.q.j
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                return LoadRenderableFromFilamentGltfTask.this.a((byte[]) obj);
            }
        }, ThreadPools.getMainExecutor());
    }
}