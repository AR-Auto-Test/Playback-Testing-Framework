package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.widget.CompoundButton;

/* compiled from: AppCompatCompoundButtonHelper.java */
/* loaded from: classes.dex */
public class i {

    /* renamed from: a  reason: collision with root package name */
    public final CompoundButton f854a;

    /* renamed from: b  reason: collision with root package name */
    public ColorStateList f855b = null;

    /* renamed from: c  reason: collision with root package name */
    public PorterDuff.Mode f856c = null;

    /* renamed from: d  reason: collision with root package name */
    public boolean f857d = false;

    /* renamed from: e  reason: collision with root package name */
    public boolean f858e = false;

    /* renamed from: f  reason: collision with root package name */
    public boolean f859f;

    public i(CompoundButton compoundButton) {
        this.f854a = compoundButton;
    }

    public void a() {
        Drawable buttonDrawable = this.f854a.getButtonDrawable();
        if (buttonDrawable != null) {
            if (this.f857d || this.f858e) {
                Drawable mutate = buttonDrawable.mutate();
                if (this.f857d) {
                    mutate.setTintList(this.f855b);
                }
                if (this.f858e) {
                    mutate.setTintMode(this.f856c);
                }
                if (mutate.isStateful()) {
                    mutate.setState(this.f854a.getDrawableState());
                }
                this.f854a.setButtonDrawable(mutate);
            }
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:18:0x0059 A[Catch: all -> 0x0063, TryCatch #1 {all -> 0x0063, blocks: (B:3:0x001c, B:5:0x0022, B:7:0x0028, B:11:0x0039, B:13:0x003f, B:15:0x0045, B:16:0x0052, B:18:0x0059, B:21:0x0065, B:23:0x006c), top: B:31:0x001c }] */
    /* JADX WARN: Removed duplicated region for block: B:23:0x006c A[Catch: all -> 0x0063, TRY_LEAVE, TryCatch #1 {all -> 0x0063, blocks: (B:3:0x001c, B:5:0x0022, B:7:0x0028, B:11:0x0039, B:13:0x003f, B:15:0x0045, B:16:0x0052, B:18:0x0059, B:21:0x0065, B:23:0x006c), top: B:31:0x001c }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void b(AttributeSet attributeSet, int i) {
        int m;
        int m2;
        Context context = this.f854a.getContext();
        int[] iArr = b.b.b.l;
        y0 r = y0.r(context, attributeSet, iArr, i, 0);
        CompoundButton compoundButton = this.f854a;
        b.j.j.q.m(compoundButton, compoundButton.getContext(), iArr, attributeSet, r.f972b, i, 0);
        boolean z = true;
        try {
            if (r.p(1) && (m2 = r.m(1, 0)) != 0) {
                try {
                    CompoundButton compoundButton2 = this.f854a;
                    compoundButton2.setButtonDrawable(b.b.d.a.a.a(compoundButton2.getContext(), m2));
                } catch (Resources.NotFoundException unused) {
                }
                if (!z && r.p(0) && (m = r.m(0, 0)) != 0) {
                    CompoundButton compoundButton3 = this.f854a;
                    compoundButton3.setButtonDrawable(b.b.d.a.a.a(compoundButton3.getContext(), m));
                }
                if (r.p(2)) {
                    this.f854a.setButtonTintList(r.c(2));
                }
                if (r.p(3)) {
                    this.f854a.setButtonTintMode(e0.c(r.j(3, -1), null));
                }
                r.f972b.recycle();
            }
            z = false;
            if (!z) {
                CompoundButton compoundButton32 = this.f854a;
                compoundButton32.setButtonDrawable(b.b.d.a.a.a(compoundButton32.getContext(), m));
            }
            if (r.p(2)) {
            }
            if (r.p(3)) {
            }
            r.f972b.recycle();
        } catch (Throwable th) {
            r.f972b.recycle();
            throw th;
        }
    }
}