package c.e.b;

import android.net.Uri;
import android.util.ArraySet;
import android.util.Log;
import android.widget.ProgressBar;
import android.widget.TextView;
import com.google.android.filament.Box;
import com.google.android.filament.gltfio.FilamentAsset;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ModelRenderable;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class bd implements c.e.b.gf.c {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Node f4578a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ vc f4579b;

    public bd(vc vcVar, Node node) {
        this.f4579b = vcVar;
        this.f4578a = node;
    }

    @Override // c.e.b.gf.c
    public void a(String str, int i, String str2) {
        ProgressBar progressBar = this.f4579b.r;
        if (progressBar != null) {
            progressBar.setProgress(i);
        }
        TextView textView = this.f4579b.s;
        if (textView != null) {
            textView.setText(i + " %");
        }
    }

    @Override // c.e.b.gf.c
    public void b(String str, String str2) {
        Objects.requireNonNull(this.f4579b);
        Log.d("LoaderARContentGroundPlaneSceneformARCore", "progress complete " + str2);
        vc vcVar = this.f4579b;
        Node node = vcVar.j;
        if (node != null) {
            node.setParent(null);
            vcVar.j = null;
        }
        CompletableFuture<ModelRenderable> build = ModelRenderable.builder().setSource(this.f4579b.f5340h, Uri.parse(str2)).setIsFilamentGltf(true).build();
        final Node node2 = this.f4578a;
        build.thenAccept(new Consumer() { // from class: c.e.b.n3
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                bd bdVar = bd.this;
                Node node3 = node2;
                Objects.requireNonNull(bdVar.f4579b);
                Log.d("LoaderARContentGroundPlaneSceneformARCore", "load3Dmodel model loaded");
                node3.setRenderable((ModelRenderable) obj);
                vc vcVar2 = bdVar.f4579b;
                Objects.requireNonNull(vcVar2);
                final ArraySet arraySet = new ArraySet();
                FilamentAsset filamentAsset = node3.getRenderableInstance().getFilamentAsset();
                if (filamentAsset.getAnimator().getAnimationCount() > 0) {
                    arraySet.add(new c.e.b.p000if.c(filamentAsset.getAnimator(), 0, Long.valueOf(System.nanoTime())));
                    vcVar2.l.getArSceneView().getScene().addOnUpdateListener(new Scene.OnUpdateListener() { // from class: c.e.b.x3
                        @Override // com.google.ar.sceneform.Scene.OnUpdateListener
                        public final void onUpdate(FrameTime frameTime) {
                            Set<c.e.b.p000if.c> set = arraySet;
                            Long valueOf = Long.valueOf(System.nanoTime());
                            for (c.e.b.p000if.c cVar : set) {
                                cVar.f4867a.applyAnimation(cVar.f4870d, ((float) ((valueOf.longValue() - cVar.f4868b.longValue()) / TimeUnit.SECONDS.toNanos(1L))) % cVar.f4869c);
                                cVar.f4867a.updateBoneMatrices();
                            }
                        }
                    });
                }
                Objects.requireNonNull(bdVar.f4579b);
                FilamentAsset filamentAsset2 = node3.getRenderableInstance().getFilamentAsset();
                if (filamentAsset2 != null) {
                    Box boundingBox = filamentAsset2.getBoundingBox();
                    float[] halfExtent = boundingBox.getHalfExtent();
                    float[] center = boundingBox.getCenter();
                    StringBuilder x = c.b.a.a.a.x("load3Dmodel center ");
                    x.append(center[0]);
                    x.append(", ");
                    x.append(center[1]);
                    x.append(", ");
                    x.append(center[2]);
                    Log.d("LoaderARContentGroundPlaneSceneformARCore", x.toString());
                    float max = 1.0f / Math.max(Math.max(halfExtent[0], halfExtent[1]), halfExtent[2]);
                    float f2 = -max;
                    node3.setLocalScale(new Vector3(f2, f2, max));
                    Log.d("LoaderARContentGroundPlaneSceneformARCore", "load3Dmodel bounds " + halfExtent[0] + ", " + halfExtent[1] + ", " + halfExtent[2] + "  Scale = " + max);
                    float f3 = center[0] * max;
                    float f4 = (center[1] * max) - (halfExtent[1] * max);
                    float f5 = center[2] * max;
                    StringBuilder x2 = c.b.a.a.a.x("load3Dmodel yCorrection ");
                    x2.append(node3.getLocalPosition());
                    x2.append(" correction = ");
                    x2.append(f3);
                    x2.append(", ");
                    x2.append(f4);
                    x2.append(", ");
                    x2.append(f5);
                    Log.d("LoaderARContentGroundPlaneSceneformARCore", x2.toString());
                    node3.setLocalPosition(new Vector3(node3.getLocalPosition().x - f3, node3.getLocalPosition().y + f4, node3.getLocalPosition().z + f5));
                }
            }
        }).exceptionally(new Function() { // from class: c.e.b.o3
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Objects.requireNonNull(bd.this.f4579b);
                StringBuilder sb = new StringBuilder();
                sb.append("load3Dmodel ");
                c.b.a.a.a.N((Throwable) obj, sb, "LoaderARContentGroundPlaneSceneformARCore");
                return null;
            }
        });
    }
}