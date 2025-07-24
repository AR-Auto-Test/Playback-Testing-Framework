package c.e.b;

import android.media.MediaPlayer;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class rc implements Node.LifecycleListener {
    public rc(jc jcVar) {
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onActivated(Node node) {
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onDeactivated(Node node) {
        MediaPlayer mediaPlayer = jc.f4941a;
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            jc.f4941a = null;
        }
        MediaPlayer mediaPlayer2 = jc.f4942b;
        if (mediaPlayer2 != null) {
            mediaPlayer2.stop();
            jc.f4942b = null;
        }
        MediaPlayer mediaPlayer3 = jc.f4943c;
        if (mediaPlayer3 != null) {
            mediaPlayer3.stop();
            jc.f4943c = null;
        }
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onUpdated(Node node, FrameTime frameTime) {
    }
}