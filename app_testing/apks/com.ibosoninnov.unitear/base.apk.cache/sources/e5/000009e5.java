package c.e.b;

import android.media.MediaPlayer;
import android.util.Log;
import android.widget.ImageView;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.SimpleTransformableNode;
import com.ibosoninnov.unitear.R;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import java.util.function.Function;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class ed implements MediaPlayer.OnPreparedListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ boolean f4694a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ float[] f4695b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ ExternalTexture f4696c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ Node f4697d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ SimpleTransformableNode f4698e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ MediaPlayer f4699f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ boolean f4700g;

    /* renamed from: h  reason: collision with root package name */
    public final /* synthetic */ String f4701h;
    public final /* synthetic */ vc i;

    public ed(vc vcVar, boolean z, float[] fArr, ExternalTexture externalTexture, Node node, SimpleTransformableNode simpleTransformableNode, MediaPlayer mediaPlayer, boolean z2, String str) {
        this.i = vcVar;
        this.f4694a = z;
        this.f4695b = fArr;
        this.f4696c = externalTexture;
        this.f4697d = node;
        this.f4698e = simpleTransformableNode;
        this.f4699f = mediaPlayer;
        this.f4700g = z2;
        this.f4701h = str;
    }

    @Override // android.media.MediaPlayer.OnPreparedListener
    public void onPrepared(MediaPlayer mediaPlayer) {
        vc vcVar = this.i;
        Node node = vcVar.j;
        if (node != null) {
            node.setParent(null);
            vcVar.j = null;
        }
        final float videoHeight = mediaPlayer.getVideoHeight();
        final float videoWidth = mediaPlayer.getVideoWidth();
        Objects.requireNonNull(this.i);
        Log.d("LoaderARContentGroundPlaneSceneformARCore", "createVideoPlayerSceneform " + videoWidth + " x " + videoHeight);
        int i = R.raw.augmented_video_material;
        if (this.f4694a) {
            i = R.raw.chroma_key_video_material;
        }
        CompletableFuture<Material> build = Material.builder().setSource(this.i.f5340h, i).build();
        final boolean z = this.f4694a;
        final float[] fArr = this.f4695b;
        final ExternalTexture externalTexture = this.f4696c;
        final Node node2 = this.f4697d;
        final SimpleTransformableNode simpleTransformableNode = this.f4698e;
        final MediaPlayer mediaPlayer2 = this.f4699f;
        final boolean z2 = this.f4700g;
        final String str = this.f4701h;
        build.thenAccept(new Consumer() { // from class: c.e.b.p3
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                boolean z3;
                Material material;
                ed edVar = ed.this;
                boolean z4 = z;
                float[] fArr2 = fArr;
                float f2 = videoWidth;
                float f3 = videoHeight;
                ExternalTexture externalTexture2 = externalTexture;
                Node node3 = node2;
                SimpleTransformableNode simpleTransformableNode2 = simpleTransformableNode;
                MediaPlayer mediaPlayer3 = mediaPlayer2;
                final boolean z5 = z2;
                String str2 = str;
                Material material2 = (Material) obj;
                Objects.requireNonNull(edVar);
                if (z4) {
                    z3 = false;
                    material = material2;
                    material2.setFloat4("keyColor", fArr2[0], fArr2[1], fArr2[2], 1.0f);
                } else {
                    z3 = false;
                    material = material2;
                }
                float f4 = f2 / f3;
                ModelRenderable makeCube = ShapeFactory.makeCube(new Vector3(f4 * 4.4f, 4.4f, 1.0E-4f), Vector3.zero(), material);
                makeCube.setShadowCaster(z3);
                makeCube.setShadowReceiver(z3);
                makeCube.getMaterial().setExternalTexture("videoTexture", externalTexture2);
                node3.setRenderable(makeCube);
                simpleTransformableNode2.setLocalScale(new Vector3(0.06f, 0.06f, 0.06f));
                final vc vcVar2 = edVar.i;
                Objects.requireNonNull(vcVar2);
                final Node node4 = new Node();
                node4.setName("playPauseButton");
                vcVar2.m.add(node4);
                node4.setParent(node3);
                node4.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.001f));
                node4.setOnTapListener(new fd(vcVar2, node4, mediaPlayer3));
                ViewRenderable.builder().setView(vcVar2.f5340h, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.s3
                    @Override // java.util.function.Consumer
                    public final void accept(Object obj2) {
                        boolean z6 = z5;
                        Node node5 = node4;
                        ViewRenderable viewRenderable = (ViewRenderable) obj2;
                        ImageView imageView = (ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view);
                        if (z6) {
                            imageView.setImageResource(R.drawable.pause);
                        } else {
                            imageView.setImageResource(R.drawable.play);
                        }
                        viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                        viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                        node5.setRenderable(viewRenderable);
                        c.b.a.a.a.C(1.0f, 1.0f, 1.0f, node5);
                    }
                }).exceptionally(new Function() { // from class: c.e.b.b4
                    @Override // java.util.function.Function
                    public final Object apply(Object obj2) {
                        Throwable th = (Throwable) obj2;
                        Objects.requireNonNull(vc.this);
                        Log.e("LoaderARContentGroundPlaneSceneformARCore", "Unable to load  renderable");
                        return null;
                    }
                });
                final vc vcVar3 = edVar.i;
                Objects.requireNonNull(vcVar3);
                final Node node5 = new Node();
                vcVar3.m.add(node5);
                node5.setParent(node3);
                node5.setLocalPosition(new Vector3(f4 * 2.0f, -2.0f, 0.001f));
                node5.setOnTapListener(new gd(vcVar3, str2));
                ViewRenderable.builder().setView(vcVar3.f5340h, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.r3
                    @Override // java.util.function.Consumer
                    public final void accept(Object obj2) {
                        Node node6 = Node.this;
                        ViewRenderable viewRenderable = (ViewRenderable) obj2;
                        ((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.fullscreen);
                        node6.setRenderable(viewRenderable);
                        c.b.a.a.a.C(0.6f, 0.6f, 0.6f, node6);
                    }
                }).exceptionally(new Function() { // from class: c.e.b.m3
                    @Override // java.util.function.Function
                    public final Object apply(Object obj2) {
                        Throwable th = (Throwable) obj2;
                        Objects.requireNonNull(vc.this);
                        Log.e("LoaderARContentGroundPlaneSceneformARCore", "Unable to load  renderable");
                        return null;
                    }
                });
                if (z5) {
                    mediaPlayer3.start();
                    vc.d(edVar.i, 10);
                    return;
                }
                mediaPlayer3.start();
                mediaPlayer3.pause();
                vc.d(edVar.i, 2000);
            }
        }).exceptionally(new Function() { // from class: c.e.b.q3
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(ed.this.i);
                Log.e("LoaderARContentGroundPlaneSceneformARCore", "Unable to load  renderable");
                return null;
            }
        });
    }
}