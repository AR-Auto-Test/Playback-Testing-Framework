package c.e.b.ef;

import android.view.View;

/* compiled from: ThumbnailAdapter.java */
/* loaded from: classes2.dex */
public class e implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f4719b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ f f4720c;

    public e(f fVar, int i) {
        this.f4720c = fVar;
        this.f4719b = i;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (this.f4720c.f4721a.get(this.f4719b).downloadStatus == -1) {
            this.f4720c.f4721a.get(this.f4719b).downloadStatus = 0;
            this.f4720c.notifyItemChanged(this.f4719b);
            f fVar = this.f4720c;
            fVar.f4723c.d(fVar.f4721a.get(this.f4719b));
        } else if ((this.f4720c.f4721a.get(this.f4719b).downloadStatus <= -1 || this.f4720c.f4721a.get(this.f4719b).downloadStatus > 100) && this.f4720c.f4721a.get(this.f4719b).downloadStatus == 101) {
            f fVar2 = this.f4720c;
            fVar2.f4723c.c(fVar2.f4721a.get(this.f4719b));
        }
    }
}