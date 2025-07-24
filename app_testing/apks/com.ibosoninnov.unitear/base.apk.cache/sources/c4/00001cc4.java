package com.google.ar.sceneform.rendering;

import android.graphics.SurfaceTexture;
import android.view.Surface;
import com.google.android.filament.Stream;
import com.google.android.filament.Texture;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class ExternalTexture {
    private static final String TAG = "ExternalTexture";
    private Stream filamentStream;
    private com.google.android.filament.Texture filamentTexture;
    private final Surface surface;
    private final SurfaceTexture surfaceTexture;

    /* loaded from: classes.dex */
    public static final class CleanupCallback implements Runnable {
        private final Stream filamentStream;
        private final com.google.android.filament.Texture filamentTexture;

        public CleanupCallback(com.google.android.filament.Texture texture, Stream stream) {
            this.filamentTexture = texture;
            this.filamentStream = stream;
        }

        @Override // java.lang.Runnable
        public void run() {
            AndroidPreconditions.checkUiThread();
            IEngine engine = EngineInstance.getEngine();
            if (engine == null || !engine.isValid()) {
                return;
            }
            com.google.android.filament.Texture texture = this.filamentTexture;
            if (texture != null) {
                engine.destroyTexture(texture);
            }
            Stream stream = this.filamentStream;
            if (stream != null) {
                engine.destroyStream(stream);
            }
        }
    }

    public ExternalTexture() {
        SurfaceTexture surfaceTexture = new SurfaceTexture(0);
        surfaceTexture.detachFromGLContext();
        this.surfaceTexture = surfaceTexture;
        this.surface = new Surface(surfaceTexture);
        initialize(new Stream.Builder().stream(surfaceTexture).build(EngineInstance.getEngine().getFilamentEngine()));
    }

    private void initialize(Stream stream) {
        if (this.filamentTexture == null) {
            IEngine engine = EngineInstance.getEngine();
            this.filamentStream = stream;
            Texture.Sampler sampler = Texture.Sampler.SAMPLER_EXTERNAL;
            com.google.android.filament.Texture build = new Texture.Builder().sampler(sampler).format(Texture.InternalFormat.RGB8).build(engine.getFilamentEngine());
            this.filamentTexture = build;
            build.setExternalStream(engine.getFilamentEngine(), stream);
            ResourceManager.getInstance().getExternalTextureCleanupRegistry().register(this, new CleanupCallback(this.filamentTexture, stream));
            return;
        }
        throw new AssertionError("Stream was initialized twice");
    }

    public Stream getFilamentStream() {
        return (Stream) Preconditions.checkNotNull(this.filamentStream);
    }

    public com.google.android.filament.Texture getFilamentTexture() {
        return (com.google.android.filament.Texture) Preconditions.checkNotNull(this.filamentTexture);
    }

    public Surface getSurface() {
        return (Surface) Preconditions.checkNotNull(this.surface);
    }

    public SurfaceTexture getSurfaceTexture() {
        return (SurfaceTexture) Preconditions.checkNotNull(this.surfaceTexture);
    }

    public ExternalTexture(int i, int i2, int i3) {
        this.surfaceTexture = null;
        this.surface = null;
        initialize(new Stream.Builder().stream(i).width(i2).height(i3).build(EngineInstance.getEngine().getFilamentEngine()));
    }
}