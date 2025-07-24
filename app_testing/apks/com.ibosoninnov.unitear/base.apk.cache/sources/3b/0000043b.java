package b.j.c.b;

import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Typeface;
import android.os.Handler;
import android.util.Log;
import android.util.TypedValue;
import java.io.IOException;
import java.lang.reflect.Method;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;

/* compiled from: ResourcesCompat.java */
/* loaded from: classes.dex */
public final class f {

    /* compiled from: ResourcesCompat.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public static final Object f2091a = new Object();

        /* renamed from: b  reason: collision with root package name */
        public static Method f2092b;

        /* renamed from: c  reason: collision with root package name */
        public static boolean f2093c;
    }

    public static Typeface a(Context context, int i) {
        if (context.isRestricted()) {
            return null;
        }
        return f(context, i, new TypedValue(), 0, null, null, false, false);
    }

    public static int b(TypedArray typedArray, XmlPullParser xmlPullParser, String str, int i, int i2) {
        return !e(xmlPullParser, str) ? i2 : typedArray.getInt(i, i2);
    }

    public static int c(TypedArray typedArray, XmlPullParser xmlPullParser, String str, int i, int i2) {
        return !e(xmlPullParser, str) ? i2 : typedArray.getResourceId(i, i2);
    }

    public static String d(TypedArray typedArray, XmlPullParser xmlPullParser, String str, int i) {
        if (e(xmlPullParser, str)) {
            return typedArray.getString(i);
        }
        return null;
    }

    public static boolean e(XmlPullParser xmlPullParser, String str) {
        return xmlPullParser.getAttributeValue("http://schemas.android.com/apk/res/android", str) != null;
    }

    /* JADX WARN: Removed duplicated region for block: B:34:0x00b8  */
    /* JADX WARN: Removed duplicated region for block: B:36:0x00bd A[ADDED_TO_REGION] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Typeface f(Context context, int i, TypedValue typedValue, int i2, e eVar, Handler handler, boolean z, boolean z2) {
        Resources resources = context.getResources();
        resources.getValue(i, typedValue, true);
        CharSequence charSequence = typedValue.string;
        if (charSequence != null) {
            String charSequence2 = charSequence.toString();
            Typeface typeface = null;
            if (charSequence2.startsWith("res/")) {
                Typeface typeface2 = b.j.d.d.f2103b.get(b.j.d.d.c(resources, i, i2));
                if (typeface2 != null) {
                    if (eVar != null) {
                        eVar.callbackSuccessAsync(typeface2, handler);
                    }
                } else if (!z2) {
                    try {
                        if (charSequence2.toLowerCase().endsWith(".xml")) {
                            b.j.c.b.a K = b.j.b.d.K(resources.getXml(i), resources);
                            if (K == null) {
                                Log.e("ResourcesCompat", "Failed to find font-family tag");
                                if (eVar != null) {
                                    eVar.callbackFailAsync(-3, handler);
                                }
                            } else {
                                typeface = b.j.d.d.a(context, K, resources, i, i2, eVar, handler, z);
                            }
                        } else {
                            typeface2 = b.j.d.d.b(context, resources, i, charSequence2, i2);
                            if (eVar != null) {
                                if (typeface2 != null) {
                                    eVar.callbackSuccessAsync(typeface2, handler);
                                } else {
                                    eVar.callbackFailAsync(-3, handler);
                                }
                            }
                        }
                    } catch (IOException e2) {
                        Log.e("ResourcesCompat", "Failed to read xml resource " + charSequence2, e2);
                        if (eVar != null) {
                            eVar.callbackFailAsync(-3, handler);
                        }
                        if (typeface == null) {
                        }
                        return typeface;
                    } catch (XmlPullParserException e3) {
                        Log.e("ResourcesCompat", "Failed to parse xml resource " + charSequence2, e3);
                        if (eVar != null) {
                        }
                        if (typeface == null) {
                        }
                        return typeface;
                    }
                }
                typeface = typeface2;
            } else if (eVar != null) {
                eVar.callbackFailAsync(-3, handler);
            }
            if (typeface == null || eVar != null || z2) {
                return typeface;
            }
            StringBuilder x = c.b.a.a.a.x("Font resource ID #0x");
            x.append(Integer.toHexString(i));
            x.append(" could not be retrieved.");
            throw new Resources.NotFoundException(x.toString());
        }
        StringBuilder x2 = c.b.a.a.a.x("Resource \"");
        x2.append(resources.getResourceName(i));
        x2.append("\" (");
        x2.append(Integer.toHexString(i));
        x2.append(") is not a Font: ");
        x2.append(typedValue);
        throw new Resources.NotFoundException(x2.toString());
    }
}