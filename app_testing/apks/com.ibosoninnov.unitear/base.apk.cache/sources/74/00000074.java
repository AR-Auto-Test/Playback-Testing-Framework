package androidx.constraintlayout.widget;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import b.h.b.i.a;
import b.h.b.i.d;
import b.h.c.b;
import b.h.c.i;

/* loaded from: classes.dex */
public class Barrier extends b {
    public int i;
    public int j;
    public a k;

    public Barrier(Context context) {
        super(context);
        super.setVisibility(8);
    }

    @Override // b.h.c.b
    public void f(AttributeSet attributeSet) {
        super.f(attributeSet);
        this.k = new a();
        if (attributeSet != null) {
            TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, i.f2010b);
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                if (index == 15) {
                    setType(obtainStyledAttributes.getInt(index, 0));
                } else if (index == 14) {
                    this.k.o0 = obtainStyledAttributes.getBoolean(index, true);
                } else if (index == 16) {
                    this.k.p0 = obtainStyledAttributes.getDimensionPixelSize(index, 0);
                }
            }
            obtainStyledAttributes.recycle();
        }
        this.f1946e = this.k;
        k();
    }

    @Override // b.h.c.b
    public void g(d dVar, boolean z) {
        int i = this.i;
        this.j = i;
        if (z) {
            if (i == 5) {
                this.j = 1;
            } else if (i == 6) {
                this.j = 0;
            }
        } else if (i == 5) {
            this.j = 0;
        } else if (i == 6) {
            this.j = 1;
        }
        if (dVar instanceof a) {
            ((a) dVar).n0 = this.j;
        }
    }

    public int getMargin() {
        return this.k.p0;
    }

    public int getType() {
        return this.i;
    }

    public void setAllowsGoneWidget(boolean z) {
        this.k.o0 = z;
    }

    public void setDpMargin(int i) {
        this.k.p0 = (int) ((i * getResources().getDisplayMetrics().density) + 0.5f);
    }

    public void setMargin(int i) {
        this.k.p0 = i;
    }

    public void setType(int i) {
        this.i = i;
    }

    public Barrier(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        super.setVisibility(8);
    }
}