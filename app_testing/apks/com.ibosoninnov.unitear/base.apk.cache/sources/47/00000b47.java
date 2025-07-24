package c.e.b;

import android.view.View;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class wc implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f5380b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ String f5381c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ vc f5382d;

    public wc(vc vcVar, int i, String str) {
        this.f5382d = vcVar;
        this.f5380b = i;
        this.f5381c = str;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        vc.c(this.f5382d, this.f5380b, this.f5381c);
    }
}