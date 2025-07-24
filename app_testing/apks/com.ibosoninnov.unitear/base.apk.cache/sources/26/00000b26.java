package c.e.b;

import android.media.MediaPlayer;
import android.util.Log;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.ibosoninnov.unitear.Player360Activity;
import com.ibosoninnov.unitear.R;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import java.util.function.Function;

/* compiled from: Player360Activity.java */
/* loaded from: classes2.dex */
public class ue implements MediaPlayer.OnPreparedListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ExternalTexture f5298a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ boolean f5299b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Player360Activity f5300c;

    public ue(Player360Activity player360Activity, ExternalTexture externalTexture, boolean z) {
        this.f5300c = player360Activity;
        this.f5298a = externalTexture;
        this.f5299b = z;
    }

    @Override // android.media.MediaPlayer.OnPreparedListener
    public void onPrepared(MediaPlayer mediaPlayer) {
        String str = this.f5300c.s;
        StringBuilder sb = new StringBuilder();
        sb.append("createVideoPlayerSceneform ");
        sb.append(mediaPlayer.getVideoWidth());
        sb.append(" x ");
        sb.append(mediaPlayer.getVideoHeight());
        Log.d(str, sb.toString());
        CompletableFuture<Material> build = Material.builder().setSource(this.f5300c, R.raw.augmented_video_material_doublesided).build();
        final ExternalTexture externalTexture = this.f5298a;
        final boolean z = this.f5299b;
        build.thenAccept(new Consumer() { // from class: c.e.b.mb
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                ue ueVar = ue.this;
                ExternalTexture externalTexture2 = externalTexture;
                boolean z2 = z;
                Objects.requireNonNull(ueVar);
                ModelRenderable makeSphere = ShapeFactory.makeSphere(0.2f, Vector3.zero(), (Material) obj);
                makeSphere.setShadowCaster(false);
                makeSphere.setShadowReceiver(false);
                makeSphere.getMaterial().setExternalTexture("videoTexture", externalTexture2);
                ueVar.f5300c.D.setRenderable(makeSphere);
                c.b.a.a.a.C(0.06f, 0.06f, 0.06f, ueVar.f5300c.D);
                ueVar.f5300c.D.setEnabled(true);
                ueVar.f5300c.x.setVisibility(8);
                if (z2) {
                    Player360Activity.r.start();
                    ueVar.f5300c.z.setImageResource(R.drawable.pause);
                }
            }
        }).exceptionally(new Function() { // from class: c.e.b.nb
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Log.e(ue.this.f5300c.s, "Unable to load  renderable");
                return null;
            }
        });
    }
}