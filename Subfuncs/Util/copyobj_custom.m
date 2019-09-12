function b = copyobj_custom(a)
% Copied from https://uk.mathworks.com/matlabcentral/answers/312653-matlab-object-assignment-copy-an-object-instead-of-creating-a-pointer
   b = eval(class(a));  %create default object of the same class as a. one valid use of eval
   for p =  properties(a).'  %copy all public properties
      try   %may fail if property is read-only
         b.(p) = a.(p);
      catch
         warning('failed to copy property: %s', p);
      end
   end
end
