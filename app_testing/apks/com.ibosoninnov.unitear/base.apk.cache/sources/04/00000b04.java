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

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class sc implements MediaPlayer.OnPreparedListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ boolean f5223a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ float[] f5224b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ ExternalTexture f5225c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ Node f5226d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ SimpleTransformableNode f5227e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ MediaPlayer f5228f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ boolean f5229g;

    /* renamed from: h  reason: collision with root package name */
    public final /* synthetic */ String f5230h;
    public final /* synthetic */ jc i;

    public sc(jc jcVar, boolean z, float[] fArr, ExternalTexture externalTexture, Node node, SimpleTransformableNode simpleTransformableNode, MediaPlayer mediaPlayer, boolean z2, String str) {
        this.i = jcVar;
        this.f5223a = z;
        this.f5224b = fArr;
        this.f5225c = externalTexture;
        this.f5226d = node;
        this.f5227e = simpleTransformableNode;
        this.f5228f = mediaPlayer;
        this.f5229g = z2;
        this.f5230h = str;
    }

    @Override // android.media.MediaPlayer.OnPreparedListener
    public void onPrepared(MediaPlayer mediaPlayer) {
        jc jcVar = this.i;
        Node node = jcVar.j;
        if (node != null) {
            node.setParent(null);
            jcVar.j = null;
        }
        final float videoHeight = mediaPlayer.getVideoHeight();
        final float videoWidth = mediaPlayer.getVideoWidth();
        Objects.requireNonNull(this.i);
        Log.d("LoaderARContentGroundPlaneSceneform", "createVideoPlayerSceneform " + videoWidth + " x " + videoHeight);
        int i = R.raw.augmented_video_material;
        if (this.f5223a) {
            i = R.raw.chroma_key_video_material;
        }
        CompletableFuture<Material> build = Material.builder().setSource(this.i.f4948h, i).build();
        final boolean z = this.f5223a;
        final float[] fArr = this.f5224b;
        final ExternalTexture externalTexture = this.f5225c;
        final Node node2 = this.f5226d;
        final SimpleTransformableNode simpleTransformableNode = this.f5227e;
        final MediaPlayer mediaPlayer2 = this.f5228f;
        final boolean z2 = this.f5229g;
        final String str = this.f5230h;
        build.thenAccept(new Consumer() { // from class: c.e.b.z2
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                boolean z3;
                Material material;
                sc scVar = sc.this;
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
                Objects.requireNonNull(scVar);
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
                final jc jcVar2 = scVar.i;
                Objects.requireNonNull(jcVar2);
                final Node node4 = new Node();
                node4.setName("playPauseButton");
                jcVar2.m.add(node4);
                node4.setParent(node3);
                node4.setLocalPosition(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.001f));
                node4.setOnTapListener(new tc(jcVar2, node4, mediaPlayer3));
                ViewRenderable.builder().setView(jcVar2.f4948h, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.k3
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
                }).exceptionally(new Function() { // from class: c.e.b.x2
                    @Override // java.util.function.Function
                    public final Object apply(Object obj2) {
                        Throwable th = (Throwable) obj2;
                        Objects.requireNonNull(jc.this);
                        Log.e("LoaderARContentGroundPlaneSceneform", "Unable to load  renderable");
                        return null;
                    }
                });
                final jc jcVar3 = scVar.i;
                Objects.requireNonNull(jcVar3);
                final Node node5 = new Node();
                jcVar3.m.add(node5);
                node5.setParent(node3);
                node5.setLocalPosition(new Vector3(f4 * 2.0f, -2.0f, 0.001f));
                node5.setOnTapListener(new uc(jcVar3, str2));
                ViewRenderable.builder().setView(jcVar3.f4948h, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.f3
                    @Override // java.util.function.Consumer
                    public final void accept(Object obj2) {
                        Node node6 = Node.this;
                        ViewRenderable viewRenderable = (ViewRenderable) obj2;
                        ((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.fullscreen);
                        node6.setRenderable(viewRenderable);
                        c.b.a.a.a.C(0.6f, 0.6f, 0.6f, node6);
                    }
                }).exceptionally(new Function() { // from class: c.e.b.d3
                    @Override // java.util.function.Function
                    public final Object apply(Object obj2) {
                        Throwable th = (Throwable) obj2;
                        Objects.requireNonNull(jc.this);
                        Log.e("LoaderARContentGroundPlaneSceneform", "Unable to load  renderable");
                        return null;
                    }
                });
                if (z5) {
                    mediaPlayer3.start();
                    jc.d(scVar.i, 10);
                    return;
                }
                jc.d(scVar.i, 2000);
            }
        }).exceptionally(new Function() { // from class: c.e.b.y2
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Objects.requireNonNull(sc.this.i);
                Log.e("LoaderARContentGroundPlaneSceneform", "Unable to load  renderable");
                return null;
            }
        });
    }
}