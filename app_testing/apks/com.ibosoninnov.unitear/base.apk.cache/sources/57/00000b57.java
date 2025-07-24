package c.e.b;

import android.view.View;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class xc implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f5412b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ String f5413c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ vc f5414d;

    public xc(vc vcVar, int i, String str) {
        this.f5414d = vcVar;
        this.f5412b = i;
        this.f5413c = str;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        vc.c(this.f5414d, this.f5412b, this.f5413c);
    }
}