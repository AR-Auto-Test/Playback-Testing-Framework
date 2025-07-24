package c.e.b;

import android.app.Activity;
import android.widget.ImageView;
import com.google.ar.sceneform.Node;
import java.util.Timer;
import java.util.TimerTask;

/* compiled from: LoaderARContentSceneformARCore.java */
/* loaded from: classes2.dex */
public class vd extends TimerTask {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Timer f5343b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int[] f5344c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ String[] f5345d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ Node f5346e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ Node f5347f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ ImageView f5348g;

    /* renamed from: h  reason: collision with root package name */
    public final /* synthetic */ yd f5349h;

    public vd(yd ydVar, Timer timer, int[] iArr, String[] strArr, Node node, Node node2, ImageView imageView) {
        this.f5349h = ydVar;
        this.f5343b = timer;
        this.f5344c = iArr;
        this.f5345d = strArr;
        this.f5346e = node;
        this.f5347f = node2;
        this.f5348g = imageView;
    }

    @Override // java.util.TimerTask, java.lang.Runnable
    public void run() {
        yd ydVar = this.f5349h;
        if (ydVar.x) {
            this.f5343b.cancel();
            return;
        }
        final int[] iArr = this.f5344c;
        iArr[0] = iArr[0] + 1;
        if (iArr[0] < 0) {
            iArr[0] = this.f5345d.length - 1;
        }
        int i = iArr[0];
        final String[] strArr = this.f5345d;
        if (i > strArr.length - 1) {
            iArr[0] = 0;
        }
        Activity activity = ydVar.f5451b;
        final Node node = this.f5346e;
        final Node node2 = this.f5347f;
        final ImageView imageView = this.f5348g;
        activity.runOnUiThread(new Runnable() { // from class: c.e.b.c8
            @Override // java.lang.Runnable
            public final void run() {
                vd vdVar = vd.this;
                String[] strArr2 = strArr;
                int[] iArr2 = iArr;
                Node node3 = node;
                Node node4 = node2;
                c.c.a.b.d(vdVar.f5349h.f5451b).k(strArr2[iArr2[0]]).C(new ud(vdVar, node3, node4)).B(imageView);
            }
        });
    }
}