package c.e.b;

import android.media.MediaPlayer;
import android.view.MotionEvent;
import android.widget.ImageView;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.ibosoninnov.unitear.R;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class fd implements Node.OnTapListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Node f4754a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ MediaPlayer f4755b;

    public fd(vc vcVar, Node node, MediaPlayer mediaPlayer) {
        this.f4754a = node;
        this.f4755b = mediaPlayer;
    }

    @Override // com.google.ar.sceneform.Node.OnTapListener
    public void onTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
        ImageView imageView = (ImageView) ((ViewRenderable) this.f4754a.getRenderable()).getView().findViewById(R.id.img_loader_view);
        if (this.f4755b.isPlaying()) {
            this.f4755b.pause();
            imageView.setImageResource(R.drawable.play);
            return;
        }
        this.f4755b.start();
        imageView.setImageResource(R.drawable.pause);
    }
}