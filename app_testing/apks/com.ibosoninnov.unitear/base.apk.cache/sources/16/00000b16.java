package c.e.b;

import android.media.MediaPlayer;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.ibosoninnov.unitear.Player360Activity;

/* compiled from: Player360Activity.java */
/* loaded from: classes2.dex */
public class te implements Node.LifecycleListener {
    public te(Player360Activity player360Activity) {
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onActivated(Node node) {
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onDeactivated(Node node) {
        MediaPlayer mediaPlayer = Player360Activity.r;
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            Player360Activity.r = null;
        }
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onUpdated(Node node, FrameTime frameTime) {
    }
}