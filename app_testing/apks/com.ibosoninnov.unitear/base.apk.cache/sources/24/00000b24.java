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

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class uc implements Node.OnTapListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ String f5293a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ jc f5294b;

    public uc(jc jcVar, String str) {
        this.f5294b = jcVar;
        this.f5293a = str;
    }

    @Override // com.google.ar.sceneform.Node.OnTapListener
    public void onTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
        Iterator<Node> it = this.f5294b.m.iterator();
        while (it.hasNext()) {
            Node next = it.next();
            if (next.getName().equals("playPauseButton")) {
                try {
                    ((ImageView) ((ViewRenderable) next.getRenderable()).getView().findViewById(R.id.img_loader_view)).setImageResource(R.drawable.play);
                } catch (Exception e2) {
                    Log.e("LoaderARContentGroundPlaneSceneform", e2.toString());
                }
            }
        }
        MediaPlayer mediaPlayer = jc.f4941a;
        if (mediaPlayer != null) {
            mediaPlayer.pause();
        }
        MediaPlayer mediaPlayer2 = jc.f4942b;
        if (mediaPlayer2 != null) {
            mediaPlayer2.pause();
        }
        MediaPlayer mediaPlayer3 = jc.f4943c;
        if (mediaPlayer3 != null) {
            mediaPlayer3.pause();
        }
        Intent intent = new Intent(this.f5294b.f4947g, VideoActivity.class);
        intent.putExtra("videoUrl", this.f5293a);
        this.f5294b.f4947g.startActivity(intent);
    }
}