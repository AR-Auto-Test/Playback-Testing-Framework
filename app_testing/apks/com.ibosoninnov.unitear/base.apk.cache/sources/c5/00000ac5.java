package c.e.b;

import android.app.Activity;
import android.widget.ImageView;
import com.google.ar.sceneform.Node;
import java.util.Timer;
import java.util.TimerTask;

/* compiled from: LoaderARContentSceneform.java */
/* loaded from: classes2.dex */
public class od extends TimerTask {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Timer f5101b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int[] f5102c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ String[] f5103d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ Node f5104e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ Node f5105f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ ImageView f5106g;

    /* renamed from: h  reason: collision with root package name */
    public final /* synthetic */ hd f5107h;

    public od(hd hdVar, Timer timer, int[] iArr, String[] strArr, Node node, Node node2, ImageView imageView) {
        this.f5107h = hdVar;
        this.f5101b = timer;
        this.f5102c = iArr;
        this.f5103d = strArr;
        this.f5104e = node;
        this.f5105f = node2;
        this.f5106g = imageView;
    }

    @Override // java.util.TimerTask, java.lang.Runnable
    public void run() {
        hd hdVar = this.f5107h;
        if (hdVar.x) {
            this.f5101b.cancel();
            return;
        }
        final int[] iArr = this.f5102c;
        iArr[0] = iArr[0] + 1;
        if (iArr[0] < 0) {
            iArr[0] = this.f5103d.length - 1;
        }
        int i = iArr[0];
        final String[] strArr = this.f5103d;
        if (i > strArr.length - 1) {
            iArr[0] = 0;
        }
        Activity activity = hdVar.f4816g;
        final Node node = this.f5104e;
        final Node node2 = this.f5105f;
        final ImageView imageView = this.f5106g;
        activity.runOnUiThread(new Runnable() { // from class: c.e.b.s4
            @Override // java.lang.Runnable
            public final void run() {
                od odVar = od.this;
                String[] strArr2 = strArr;
                int[] iArr2 = iArr;
                Node node3 = node;
                Node node4 = node2;
                c.c.a.b.d(odVar.f5107h.f4816g).k(strArr2[iArr2[0]]).C(new nd(odVar, node3, node4)).B(imageView);
            }
        });
    }
}