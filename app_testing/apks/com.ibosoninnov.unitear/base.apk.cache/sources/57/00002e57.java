package pl.droidsonroids.gif;

import android.content.Context;
import android.net.Uri;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.widget.ImageView;
import h.a.a.c;
import h.a.a.f;
import h.a.a.g;
import java.io.IOException;
import java.util.List;

/* loaded from: classes2.dex */
public class GifImageView extends ImageView {

    /* renamed from: b  reason: collision with root package name */
    public boolean f6266b;

    public GifImageView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        g.a aVar;
        List<String> list = g.f6245a;
        if (attributeSet != null && !isInEditMode()) {
            aVar = new g.a(this, attributeSet, 0, 0);
            int i = aVar.f6249b;
            if (i >= 0) {
                g.a(i, getDrawable());
                g.a(i, getBackground());
            }
        } else {
            aVar = new g.a();
        }
        this.f6266b = aVar.f6248a;
        int i2 = aVar.f6246c;
        if (i2 > 0) {
            super.setImageResource(i2);
        }
        int i3 = aVar.f6247d;
        if (i3 > 0) {
            super.setBackgroundResource(i3);
        }
    }

    @Override // android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (!(parcelable instanceof f)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        f fVar = (f) parcelable;
        super.onRestoreInstanceState(fVar.getSuperState());
        fVar.a(getDrawable(), 0);
        fVar.a(getBackground(), 1);
    }

    @Override // android.view.View
    public Parcelable onSaveInstanceState() {
        return new f(super.onSaveInstanceState(), this.f6266b ? getDrawable() : null, this.f6266b ? getBackground() : null);
    }

    @Override // android.view.View
    public void setBackgroundResource(int i) {
        if (g.b(this, false, i)) {
            return;
        }
        super.setBackgroundResource(i);
    }

    public void setFreezesAnimation(boolean z) {
        this.f6266b = z;
    }

    @Override // android.widget.ImageView
    public void setImageResource(int i) {
        if (g.b(this, true, i)) {
            return;
        }
        super.setImageResource(i);
    }

    /* JADX WARN: Removed duplicated region for block: B:13:? A[RETURN, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:8:0x0019  */
    @Override // android.widget.ImageView
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void setImageURI(Uri uri) {
        boolean z;
        List<String> list = g.f6245a;
        if (uri != null) {
            try {
                setImageDrawable(new c(getContext().getContentResolver(), uri));
                z = true;
            } catch (IOException unused) {
            }
            if (z) {
                super.setImageURI(uri);
                return;
            }
            return;
        }
        z = false;
        if (z) {
        }
    }
}