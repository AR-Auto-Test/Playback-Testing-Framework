package c.e.b;

import android.media.MediaPlayer;
import android.view.MotionEvent;
import android.widget.ImageView;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.ibosoninnov.unitear.R;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class tc implements Node.OnTapListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Node f5257a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ MediaPlayer f5258b;

    public tc(jc jcVar, Node node, MediaPlayer mediaPlayer) {
        this.f5257a = node;
        this.f5258b = mediaPlayer;
    }

    @Override // com.google.ar.sceneform.Node.OnTapListener
    public void onTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
        ImageView imageView = (ImageView) ((ViewRenderable) this.f5257a.getRenderable()).getView().findViewById(R.id.img_loader_view);
        if (this.f5258b.isPlaying()) {
            this.f5258b.pause();
            imageView.setImageResource(R.drawable.play);
            return;
        }
        this.f5258b.start();
        imageView.setImageResource(R.drawable.pause);
    }
}