package c.e.b.p000if;

import android.app.Activity;
import android.util.Log;
import android.widget.ImageView;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.SceneView;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.ibosoninnov.unitear.R;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;

/* compiled from: LogoNode.java */
/* renamed from: c.e.b.if.l  reason: invalid package */
/* loaded from: classes2.dex */
public class l {

    /* renamed from: a  reason: collision with root package name */
    public Node f4891a;

    public l(Activity activity, SceneView sceneView) {
        final float width = sceneView.getWidth();
        final float height = sceneView.getHeight();
        Node node = new Node();
        this.f4891a = node;
        node.setParent(sceneView.getScene().getCamera());
        final Vector3 forward = sceneView.getScene().getCamera().getForward();
        float verticalFovDegrees = sceneView.getScene().getCamera().getVerticalFovDegrees() / 60.0f;
        this.f4891a.setLocalPosition(new Vector3((width / height) * verticalFovDegrees * 0.18f, (height / width) * verticalFovDegrees * 0.13f, -0.5f));
        this.f4891a.setLookDirection(forward, sceneView.getScene().getCamera().getUp());
        ViewRenderable.builder().setView(activity, R.layout.img_view).build().thenAccept(new Consumer() { // from class: c.e.b.if.a
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                l lVar = l.this;
                Vector3 vector3 = forward;
                float f2 = width;
                float f3 = height;
                ViewRenderable viewRenderable = (ViewRenderable) obj;
                Objects.requireNonNull(lVar);
                ((ImageView) viewRenderable.getView().findViewById(R.id.img_loader_view)).setImageResource(2131165506);
                viewRenderable.setShadowCaster(false);
                viewRenderable.setShadowReceiver(false);
                viewRenderable.setHorizontalAlignment(ViewRenderable.HorizontalAlignment.CENTER);
                viewRenderable.setVerticalAlignment(ViewRenderable.VerticalAlignment.CENTER);
                lVar.f4891a.setRenderable(viewRenderable);
                lVar.f4891a.setLocalScale(new Vector3(0.12f, 0.12f, 0.12f));
                Log.d("LogoNode", "Created fwd = " + vector3 + " " + f2 + " x " + f3);
            }
        }).exceptionally((Function<Throwable, ? extends Void>) b.f4866a);
    }

    public void a() {
        Node node = this.f4891a;
        if (node != null) {
            node.setParent(null);
            Log.d("LogoNode", "Destroyed");
        }
    }
}