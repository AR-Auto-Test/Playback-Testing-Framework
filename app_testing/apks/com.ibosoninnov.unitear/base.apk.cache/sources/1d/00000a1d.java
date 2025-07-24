package c.e.b;

import android.content.Intent;
import android.media.MediaPlayer;
import android.util.Log;
import android.view.MotionEvent;
import android.widget.ImageView;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.VideoActivity;
import java.util.Iterator;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class gd implements Node.OnTapListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ String f4785a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ vc f4786b;

    public gd(vc vcVar, String str) {
        this.f4786b = vcVar;
        this.f4785a = str;
    }

    @Override // com.google.ar.sceneform.Node.OnTapListener
    public void onTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
        Iterator<Node> it = this.f4786b.m.iterator();
        while (it.hasNext()) {
            Node next = it.next();
            if (next.getName().equals("playPauseButton")) {
                try {
                    ((ImageView) ((ViewRenderable) next.getRenderable()).getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.play);
                } catch (Exception e2) {
                    Log.e("LoaderARContentGroundPlaneSceneformARCore", e2.toString());
                }
            }
        }
        MediaPlayer mediaPlayer = vc.f5333a;
        if (mediaPlayer != null) {
            mediaPlayer.pause();
        }
        MediaPlayer mediaPlayer2 = vc.f5334b;
        if (mediaPlayer2 != null) {
            mediaPlayer2.pause();
        }
        MediaPlayer mediaPlayer3 = vc.f5335c;
        if (mediaPlayer3 != null) {
            mediaPlayer3.pause();
        }
        Intent intent = new Intent(this.f4786b.f5339g, VideoActivity.class);
        intent.putExtra("videoUrl", this.f4785a);
        this.f4786b.f5339g.startActivity(intent);
    }
}