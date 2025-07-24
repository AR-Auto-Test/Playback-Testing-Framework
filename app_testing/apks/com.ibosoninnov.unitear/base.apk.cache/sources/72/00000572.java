package b.v;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.util.Log;
import b.v.q;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/* compiled from: ActivityNavigator.java */
@q.b("activity")
/* loaded from: classes.dex */
public class a extends q<C0050a> {

    /* renamed from: a  reason: collision with root package name */
    public Context f2609a;

    /* renamed from: b  reason: collision with root package name */
    public Activity f2610b;

    /* compiled from: ActivityNavigator.java */
    /* renamed from: b.v.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0050a extends j {
        public Intent j;
        public String k;

        public C0050a(q<? extends C0050a> qVar) {
            super(qVar);
        }

        @Override // b.v.j
        public void d(Context context, AttributeSet attributeSet) {
            super.d(context, attributeSet);
            TypedArray obtainAttributes = context.getResources().obtainAttributes(attributeSet, s.f2679a);
            String string = obtainAttributes.getString(4);
            if (string != null) {
                string = string.replace("${applicationId}", context.getPackageName());
            }
            if (this.j == null) {
                this.j = new Intent();
            }
            this.j.setPackage(string);
            String string2 = obtainAttributes.getString(0);
            if (string2 != null) {
                if (string2.charAt(0) == '.') {
                    string2 = context.getPackageName() + string2;
                }
                ComponentName componentName = new ComponentName(context, string2);
                if (this.j == null) {
                    this.j = new Intent();
                }
                this.j.setComponent(componentName);
            }
            String string3 = obtainAttributes.getString(1);
            if (this.j == null) {
                this.j = new Intent();
            }
            this.j.setAction(string3);
            String string4 = obtainAttributes.getString(2);
            if (string4 != null) {
                Uri parse = Uri.parse(string4);
                if (this.j == null) {
                    this.j = new Intent();
                }
                this.j.setData(parse);
            }
            this.k = obtainAttributes.getString(3);
            obtainAttributes.recycle();
        }

        @Override // b.v.j
        public String toString() {
            Intent intent = this.j;
            ComponentName component = intent == null ? null : intent.getComponent();
            StringBuilder sb = new StringBuilder();
            sb.append(super.toString());
            if (component != null) {
                sb.append(" class=");
                sb.append(component.getClassName());
            } else {
                Intent intent2 = this.j;
                String action = intent2 != null ? intent2.getAction() : null;
                if (action != null) {
                    sb.append(" action=");
                    sb.append(action);
                }
            }
            return sb.toString();
        }
    }

    /* compiled from: ActivityNavigator.java */
    /* loaded from: classes.dex */
    public static final class b implements q.a {
    }

    public a(Context context) {
        this.f2609a = context;
        while (context instanceof ContextWrapper) {
            if (context instanceof Activity) {
                this.f2610b = (Activity) context;
                return;
            }
            context = ((ContextWrapper) context).getBaseContext();
        }
    }

    /* JADX DEBUG: Return type fixed from 'b.v.j' to match base method */
    @Override // b.v.q
    public C0050a a() {
        return new C0050a(this);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [b.v.j, android.os.Bundle, b.v.o, b.v.q$a] */
    @Override // b.v.q
    public j b(C0050a c0050a, Bundle bundle, o oVar, q.a aVar) {
        Intent intent;
        int intExtra;
        C0050a c0050a2 = c0050a;
        if (c0050a2.j != null) {
            Intent intent2 = new Intent(c0050a2.j);
            if (bundle != null) {
                intent2.putExtras(bundle);
                String str = c0050a2.k;
                if (!TextUtils.isEmpty(str)) {
                    StringBuffer stringBuffer = new StringBuffer();
                    Matcher matcher = Pattern.compile("\\{(.+?)\\}").matcher(str);
                    while (matcher.find()) {
                        String group = matcher.group(1);
                        if (bundle.containsKey(group)) {
                            matcher.appendReplacement(stringBuffer, "");
                            stringBuffer.append(Uri.encode(bundle.get(group).toString()));
                        } else {
                            throw new IllegalArgumentException("Could not find " + group + " in " + bundle + " to fill data pattern " + str);
                        }
                    }
                    matcher.appendTail(stringBuffer);
                    intent2.setData(Uri.parse(stringBuffer.toString()));
                }
            }
            boolean z = aVar instanceof b;
            if (z) {
                Objects.requireNonNull((b) aVar);
                intent2.addFlags(0);
            }
            if (!(this.f2609a instanceof Activity)) {
                intent2.addFlags(268435456);
            }
            if (oVar != null && oVar.f2662a) {
                intent2.addFlags(536870912);
            }
            Activity activity = this.f2610b;
            if (activity != null && (intent = activity.getIntent()) != null && (intExtra = intent.getIntExtra("android-support-navigation:ActivityNavigator:current", 0)) != 0) {
                intent2.putExtra("android-support-navigation:ActivityNavigator:source", intExtra);
            }
            intent2.putExtra("android-support-navigation:ActivityNavigator:current", c0050a2.f2645d);
            Resources resources = this.f2609a.getResources();
            if (oVar != null) {
                int i = oVar.f2667f;
                int i2 = oVar.f2668g;
                if ((i > 0 && resources.getResourceTypeName(i).equals("animator")) || (i2 > 0 && resources.getResourceTypeName(i2).equals("animator"))) {
                    StringBuilder x = c.b.a.a.a.x("Activity destinations do not support Animator resource. Ignoring popEnter resource ");
                    x.append(resources.getResourceName(i));
                    x.append(" and popExit resource ");
                    x.append(resources.getResourceName(i2));
                    x.append("when launching ");
                    x.append(c0050a2);
                    Log.w("ActivityNavigator", x.toString());
                } else {
                    intent2.putExtra("android-support-navigation:ActivityNavigator:popEnterAnim", i);
                    intent2.putExtra("android-support-navigation:ActivityNavigator:popExitAnim", i2);
                }
            }
            if (z) {
                Objects.requireNonNull((b) aVar);
                this.f2609a.startActivity(intent2);
            } else {
                this.f2609a.startActivity(intent2);
            }
            if (oVar == null || this.f2610b == null) {
                return null;
            }
            int i3 = oVar.f2665d;
            int i4 = oVar.f2666e;
            if ((i3 <= 0 || !resources.getResourceTypeName(i3).equals("animator")) && (i4 <= 0 || !resources.getResourceTypeName(i4).equals("animator"))) {
                if (i3 >= 0 || i4 >= 0) {
                    this.f2610b.overridePendingTransition(Math.max(i3, 0), Math.max(i4, 0));
                    return null;
                }
                return null;
            }
            StringBuilder x2 = c.b.a.a.a.x("Activity destinations do not support Animator resource. Ignoring enter resource ");
            x2.append(resources.getResourceName(i3));
            x2.append(" and exit resource ");
            x2.append(resources.getResourceName(i4));
            x2.append("when launching ");
            x2.append(c0050a2);
            Log.w("ActivityNavigator", x2.toString());
            return null;
        }
        throw new IllegalStateException(c.b.a.a.a.s(c.b.a.a.a.x("Destination "), c0050a2.f2645d, " does not have an Intent set."));
    }

    @Override // b.v.q
    public boolean e() {
        Activity activity = this.f2610b;
        if (activity != null) {
            activity.finish();
            return true;
        }
        return false;
    }
}