package b.h.c;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.TypedValue;
import android.util.Xml;
import b.d.b.m0;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.HashMap;
import org.xmlpull.v1.XmlPullParser;

/* compiled from: ConstraintAttribute.java */
/* loaded from: classes.dex */
public class a {

    /* renamed from: a  reason: collision with root package name */
    public String f1936a;

    /* renamed from: b  reason: collision with root package name */
    public int f1937b;

    /* renamed from: c  reason: collision with root package name */
    public int f1938c;

    /* renamed from: d  reason: collision with root package name */
    public float f1939d;

    /* renamed from: e  reason: collision with root package name */
    public String f1940e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f1941f;

    /* renamed from: g  reason: collision with root package name */
    public int f1942g;

    public a(String str, int i, Object obj) {
        this.f1936a = str;
        this.f1937b = i;
        b(obj);
    }

    public static void a(Context context, XmlPullParser xmlPullParser, HashMap<String, a> hashMap) {
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(Xml.asAttributeSet(xmlPullParser), i.f2012d);
        int indexCount = obtainStyledAttributes.getIndexCount();
        String str = null;
        int i = 0;
        Object obj = null;
        for (int i2 = 0; i2 < indexCount; i2++) {
            int index = obtainStyledAttributes.getIndex(i2);
            if (index == 0) {
                str = obtainStyledAttributes.getString(index);
                if (str != null && str.length() > 0) {
                    str = Character.toUpperCase(str.charAt(0)) + str.substring(1);
                }
            } else if (index == 1) {
                obj = Boolean.valueOf(obtainStyledAttributes.getBoolean(index, false));
                i = 6;
            } else if (index == 3) {
                obj = Integer.valueOf(obtainStyledAttributes.getColor(index, 0));
                i = 3;
            } else if (index == 2) {
                obj = Integer.valueOf(obtainStyledAttributes.getColor(index, 0));
                i = 4;
            } else {
                if (index == 7) {
                    obj = Float.valueOf(TypedValue.applyDimension(1, obtainStyledAttributes.getDimension(index, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD), context.getResources().getDisplayMetrics()));
                } else if (index == 4) {
                    obj = Float.valueOf(obtainStyledAttributes.getDimension(index, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
                } else if (index == 5) {
                    obj = Float.valueOf(obtainStyledAttributes.getFloat(index, Float.NaN));
                    i = 2;
                } else if (index == 6) {
                    obj = Integer.valueOf(obtainStyledAttributes.getInteger(index, -1));
                    i = 1;
                } else if (index == 8) {
                    obj = obtainStyledAttributes.getString(index);
                    i = 5;
                }
                i = 7;
            }
        }
        if (str != null && obj != null) {
            hashMap.put(str, new a(str, i, obj));
        }
        obtainStyledAttributes.recycle();
    }

    public void b(Object obj) {
        switch (m0.f(this.f1937b)) {
            case 0:
                this.f1938c = ((Integer) obj).intValue();
                return;
            case 1:
                this.f1939d = ((Float) obj).floatValue();
                return;
            case 2:
            case 3:
                this.f1942g = ((Integer) obj).intValue();
                return;
            case 4:
                this.f1940e = (String) obj;
                return;
            case 5:
                this.f1941f = ((Boolean) obj).booleanValue();
                return;
            case 6:
                this.f1939d = ((Float) obj).floatValue();
                return;
            default:
                return;
        }
    }

    public a(a aVar, Object obj) {
        this.f1936a = aVar.f1936a;
        this.f1937b = aVar.f1937b;
        b(obj);
    }
}