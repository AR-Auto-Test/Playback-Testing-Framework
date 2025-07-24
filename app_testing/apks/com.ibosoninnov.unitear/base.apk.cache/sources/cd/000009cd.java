package c.e.b;

import android.media.MediaPlayer;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class dd implements Node.LifecycleListener {
    public dd(vc vcVar) {
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onActivated(Node node) {
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onDeactivated(Node node) {
        MediaPlayer mediaPlayer = vc.f5333a;
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            vc.f5333a = null;
        }
        MediaPlayer mediaPlayer2 = vc.f5334b;
        if (mediaPlayer2 != null) {
            mediaPlayer2.stop();
            vc.f5334b = null;
        }
        MediaPlayer mediaPlayer3 = vc.f5335c;
        if (mediaPlayer3 != null) {
            mediaPlayer3.stop();
            vc.f5335c = null;
        }
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onUpdated(Node node, FrameTime frameTime) {
    }
}